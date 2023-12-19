from __future__ import absolute_import
from __future__ import print_function
import numpy as np

from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, concatenate, Dropout
from keras.optimizers import RMSprop
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras import regularizers

import matplotlib.pyplot as plt
import argparse

import time

from scipy.sparse import vstack, csc_matrix
from sklearn.model_selection import train_test_split
from utils_new import dataLoading, aucPerformance, writeResults, get_data_from_svmlight_file, dataLoading_noheader
from data_interpolation import inject_noise, inject_noise_sparse
# from prenet import reg_network, reg_network_no_feature_learner, reg_network_deeper, load_model_weight_predict, pair_generator
from prenet_utils import *


def prenet_predict(x_train, x_test, y_train, model_name="baselines/prenet", known_outliers=60, cont_rate=0.02,
                   epochs=50, batch_size=512, nb_batch=20, network_depth=2, data_format=0):
    outlier_indices = np.where(y_train == 1)[0]
    n_outliers = len(outlier_indices)
    print("Original training size: %d, No. outliers: %d" % (x_train.shape[0], n_outliers))

    # TODO maybe we need to return the following to its previous state (it used to be labels instead of y_train)
    # THIS USED TO BE A LEAK
    n_noise = len(np.where(y_train == 0)[0]) * cont_rate / (1. - cont_rate)
    n_noise = int(n_noise)

    # TODO maybe we need to return the following to its previous state (it used to be labels instead of y_train,
    #  and x instead of x_train)
    # THIS USED TO BE A LEAK
    outlier_indices = np.where(y_train == 1)[0]
    outliers = x_train[outlier_indices]
    n_outliers_org = outliers.shape[0]

    rng = np.random.RandomState(42)
    if data_format == 0:  # todo check which data format we should use (need to ask max)
        if n_outliers > known_outliers:
            mn = n_outliers - known_outliers
            remove_idx = rng.choice(outlier_indices, mn, replace=False)
            x_train = np.delete(x_train, remove_idx, axis=0)
            y_train = np.delete(y_train, remove_idx, axis=0)

        noises = inject_noise(outliers, n_noise)
        x_train = np.append(x_train, noises, axis=0)
        y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))

    else:
        if n_outliers > known_outliers:
            mn = n_outliers - known_outliers
            remove_idx = rng.choice(outlier_indices, mn, replace=False)
            retain_idx = set(np.arange(x_train.shape[0])) - set(remove_idx)
            retain_idx = list(retain_idx)
            x_train = x_train[retain_idx]
            y_train = y_train[retain_idx]

        noises = inject_noise_sparse(outliers, n_noise)
        x_train = vstack([x_train, noises])
        y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))

    outlier_indices = np.where(y_train == 1)[0]
    inlier_indices = np.where(y_train == 0)[0]
    print(y_train.shape[0], outlier_indices.shape[0], inlier_indices.shape[0], n_noise)
    n_samples_trn = x_train.shape[0]
    input_shape = x_train.shape[1:]
    n_outliers = len(outlier_indices)
    print("Training data size: %d, No. outliers: %d" % (x_train.shape[0], n_outliers))
    Y = np.zeros(x_train.shape[0])
    Y[outlier_indices] = 1

    if network_depth == 2:
        model = reg_network(input_shape)
    elif network_depth == 1:
        model = reg_network_no_feature_learner(input_shape)
    else:
        model = reg_network_deeper(input_shape)

    checkpointer = ModelCheckpoint(model_name, monitor='loss', verbose=0,
                                   save_best_only=True, save_weights_only=True)
    history = model.fit_generator(
        pair_generator(x_train, outlier_indices, inlier_indices, Y, batch_size, nb_batch, rng),
        steps_per_epoch=nb_batch,
        epochs=epochs,
        callbacks=[checkpointer])

    scores = load_model_weight_predict(model_name, input_shape, network_depth,
                                       x_test, x_train[inlier_indices], x_train[outlier_indices])
    return [scores, n_outliers_org, n_samples_trn]

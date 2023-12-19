from __future__ import absolute_import
from __future__ import print_function

import os

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

MAX_INT = np.iinfo(np.int32).max

data_format = 0

uu = 0;
au = 4;
aa = 8
# uu=0; au=1; aa=2

ensemble_size = 30

h_lambda = 0.1


def regression_loss(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


def pair_generator(x, outlier_indices, inlier_indices, Y, batch_size, nb_batch, rng):
    """batch generator
    """
    rng = np.random.RandomState(rng.randint(MAX_INT, size=1))
    counter = 0
    while 1:
        if data_format == 0:
            samples1, samples2, training_labels = pair_batch_generation(x, outlier_indices, inlier_indices, Y,
                                                                        batch_size, rng)
        else:
            samples1, samples2, training_labels = pair_batch_generation_sparse(x, outlier_indices, inlier_indices,
                                                                               batch_size, rng)
        counter += 1
        yield ([samples1, samples2], training_labels)
        if (counter > nb_batch):
            counter = 0


def pair_batch_generation(x_train, outlier_indices, inlier_indices, Y, batch_size, rng):
    '''batchs of samples.
    Alternates between positive and negative pairs.
    '''
    dim = x_train.shape[1]
    pairs1 = np.empty((batch_size, dim))
    pairs2 = np.empty((batch_size, dim))
    labels = []
    n_inliers = len(inlier_indices)
    n_outliers = len(outlier_indices)

    block_size = int(batch_size / 4)
    sid = rng.choice(n_inliers, block_size * 4, replace=False)
    pairs1[0:block_size * 2] = x_train[inlier_indices[sid[0:block_size * 2]]]
    pairs2[0:block_size * 2] = x_train[inlier_indices[sid[block_size * 2:block_size * 4]]]
    labels += 2 * block_size * [uu]

    sid = rng.choice(n_inliers, block_size, replace=False)
    pairs1[block_size * 2:block_size * 3] = x_train[inlier_indices[sid]]
    sid = rng.choice(n_outliers, block_size)
    pairs2[block_size * 2:block_size * 3] = x_train[outlier_indices[sid]]
    labels += block_size * [au]

    for i in np.arange(block_size * 3, batch_size):
        sid = rng.choice(n_outliers, 2, replace=False)
        z1 = x_train[outlier_indices[sid[0]]]
        z2 = x_train[outlier_indices[sid[1]]]
        pairs1[i] = z1
        pairs2[i] = z2
        labels += [aa]

    return pairs1, pairs2, np.array(labels).astype(float)


def pair_batch_generation_sparse(x_train, outlier_indices, inlier_indices, batch_size, rng):
    '''batchs of samples.
    Alternates between positive and negative pairs.
    '''
    pairs1 = np.empty((batch_size))
    pairs2 = np.empty((batch_size))
    labels = []
    n_inliers = len(inlier_indices)
    n_outliers = len(outlier_indices)
    j = 0
    for i in range(batch_size):
        if i % 2 == 0:
            sid = rng.choice(n_inliers, 2, replace=False)
            z1 = inlier_indices[sid[0]]
            z2 = inlier_indices[sid[1]]
            pairs1[i] = z1
            pairs2[i] = z2
            labels += [uu]
        else:
            if j % 2 == 0:
                sid = rng.choice(n_inliers, 1)
                z1 = inlier_indices[sid]
                sid = rng.choice(n_outliers, 1)
                z2 = outlier_indices[sid]
                pairs1[i] = z1[0]
                pairs2[i] = z2[0]
                labels += [au]

            else:
                sid = rng.choice(n_outliers, 2, replace=False)
                z1 = outlier_indices[sid[0]]
                z2 = outlier_indices[sid[1]]
                pairs1[i] = z1
                pairs2[i] = z2
                labels += [aa]
            j += 1
    pairs1 = x_train[pairs1, :].toarray()
    pairs2 = x_train[pairs2, :].toarray()
    return pairs1, pairs2, np.array(labels).astype(float)


def reg_network_deeper(input_shape):
    x_input = Input(shape=input_shape)
    intermediate = Dense(1000, activation='relu',
                         kernel_regularizer=regularizers.l2(h_lambda), name='hl1')(x_input)
    intermediate = Dense(250, activation='relu',
                         kernel_regularizer=regularizers.l2(h_lambda), name='hl2')(intermediate)
    intermediate = Dense(20, activation='relu',
                         kernel_regularizer=regularizers.l2(h_lambda), name='hl3')(intermediate)
    base_network = Model(x_input, intermediate)
    print(base_network.summary())

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    input_merge = concatenate([processed_a, processed_b])
    anomaly_score = Dense(1, activation='linear', name='score')(input_merge)

    model = Model([input_a, input_b], anomaly_score)
    #    print(model.summary())

    rms = RMSprop(clipnorm=1.)
    model.compile(loss=regression_loss, optimizer=rms)
    return model


def reg_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    x_input = Input(shape=input_shape)
    intermediate = Dense(20, activation='relu',
                         kernel_regularizer=regularizers.l2(h_lambda), name='hl1')(x_input)
    base_network = Model(x_input, intermediate)
    #    print(base_network.summary())

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    input_merge = concatenate([processed_a, processed_b])
    # input_merge = Dense(20, activation='relu',  name = 'interaction')(input_merge)
    anomaly_score = Dense(1, activation='linear', name='score')(input_merge)

    model = Model([input_a, input_b], anomaly_score)
    #    print(model.summary())

    rms = RMSprop(clipnorm=1.)
    model.compile(loss=regression_loss, optimizer=rms)
    return model


def reg_network_no_feature_learner(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    input_merge = concatenate([input_a, input_b])
    anomaly_score = Dense(1, activation='linear', name='score')(input_merge)

    model = Model([input_a, input_b], anomaly_score)
    print(model.summary())

    rms = RMSprop(clipnorm=1.)
    model.compile(loss=regression_loss, optimizer=rms)
    return model


def load_model_weight_predict(model_name, input_shape, network_depth, x_test, inliers, outliers):
    if network_depth == 2:
        model = reg_network(input_shape)
    elif network_depth == 1:
        model = reg_network_no_feature_learner(input_shape)
    else:
        model = reg_network_deeper(input_shape)
    model.load_weights(model_name)
    scoring = Model(inputs=model.input, outputs=model.output)

    if data_format == 0:
        runs = ensemble_size
        rng = np.random.RandomState(42)
        test_size = x_test.shape[0]
        scores = np.zeros((test_size, runs))
        n_sample = inliers.shape[0]
        for i in np.arange(runs):
            idx = rng.choice(n_sample, 1)
            obj = inliers[idx]
            inlier_seed = np.tile(obj, (test_size, 1))
            scores[:, i] = scoring.predict([inlier_seed, x_test]).flatten()
        mean_score = np.mean(scores, axis=1)

        runs = ensemble_size
        rng = np.random.RandomState(42)
        test_size = x_test.shape[0]
        scores = np.zeros((test_size, runs))
        n_sample = outliers.shape[0]
        for i in np.arange(runs):
            idx = rng.choice(n_sample, 1)
            obj = outliers[idx]
            outlier_seed = np.tile(obj, (test_size, 1))
            scores[:, i] = scoring.predict([x_test, outlier_seed]).flatten()
        mean_score += np.mean(scores, axis=1)

        scores = mean_score / 2
    else:
        data_size = x_test.shape[0]
        count = 512
        if count > data_size:
            count = data_size

        runs = ensemble_size
        scores_a = np.zeros((data_size, runs))
        scores_u = np.zeros((data_size, runs))

        i = 0
        while i < data_size:
            subset = x_test[i:count].toarray()
            rng = np.random.RandomState(42)
            n_sample = inliers.shape[0]
            for j in np.arange(runs):
                idx = rng.choice(n_sample, 1)
                obj = inliers[idx].toarray()
                inlier_seed = np.tile(obj, (count - i, 1))
                scores_u[i:count, j] = scoring.predict([inlier_seed, subset]).flatten()

            rng = np.random.RandomState(42)
            n_sample = outliers.shape[0]
            for j in np.arange(runs):
                idx = rng.choice(n_sample, 1)
                obj = outliers[idx].toarray()
                outlier_seed = np.tile(obj, (count - i, 1))
                scores_a[i:count, j] = scoring.predict([subset, outlier_seed]).flatten()

            if i % 1024 == 0:
                print(i)
            i = count
            count += 512
            if count > data_size:
                count = data_size

        assert count == data_size
        mean_score = np.mean(scores_u, axis=1)
        mean_score += np.mean(scores_a, axis=1)
        scores = mean_score / 2

    return scores


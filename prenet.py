"""
@author: Guansong Pang

Source code for the PReNet algorithm in KDD'23.
"""

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
from prenet_predict import prenet_predict
from prenet_utils import *

MAX_INT = np.iinfo(np.int32).max

data_format = 0

uu = 0;
au = 4;
aa = 8
# uu=0; au=1; aa=2

ensemble_size = 30

h_lambda = 0.1


def run_prenet(args):
    names = args.data_set.split(',')
    network_depth = int(args.network_depth)
    for nm in names:
        runs = args.runs
        rauc = np.zeros(runs)
        ap = np.zeros(runs)
        filename = nm.strip()
        n_outliers = 0
        global data_format
        data_format = int(args.data_format)
        if data_format == 0:
            x, labels = dataLoading(args.input_path + filename + ".csv")
        else:
            x, labels = get_data_from_svmlight_file(args.input_path + filename + ".svm")
            x = x.tocsr()


        train_time = 0
        test_time = 0
        global h_lambda
        h_lambda = float(args.h_lambda)
        global uu, au, aa
        ordinal_labels = args.ordinal_labels
        ordinal_labels = ordinal_labels.split(',')
        uu = float(ordinal_labels[0])
        au = float(ordinal_labels[1])
        aa = float(ordinal_labels[2])
        global ensemble_size
        ensemble_size = args.ensemble_size

        for i in np.arange(runs):
            x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42,
                                                                stratify=labels)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            print(filename + ': round ' + str(i))


            # Input: x_train, x_test, y_train, y_test
            # | Output scores

            input_shape = x_train.shape[1:]
            start_time = time.time()

            model_name = "./model/prenet_" + filename + "_" + str(args.cont_rate) + "cr_" + str(args.batch_size) +\
                         "bs_" + str(args.known_outliers) + "ko_" + str(network_depth) + "d.h5"

            scores, n_outliers_org, n_samples_trn = prenet_predict(x_train, x_test, y_train, model_name,
                                                            args.known_outliers, args.cont_rate,
                                                            args.epochs, args.batch_size, args.nb_batch, network_depth,
                                                            data_format)
            train_time += time.time() - start_time

            rauc[i], ap[i] = aucPerformance(scores, y_test)

        mean_auc = np.mean(rauc)
        std_auc = np.std(rauc)
        mean_aucpr = np.mean(ap)
        std_aucpr = np.std(ap)
        train_time = train_time / runs
        test_time = test_time / runs
        print("average AUC-ROC: %.4f, average AUC-PR: %.4f" % (mean_auc, mean_aucpr))
        print("average runtime: %.4f seconds" % (train_time + test_time))
        ordinal_labels = str(h_lambda) + ":" + str(uu) + "_" + str(au) + "_" + str(aa)
        writeResults(ordinal_labels + "_" + filename + '_' + str(network_depth), x.shape[0], x.shape[1], n_samples_trn,
                     n_outliers_org, n_outliers,
                     network_depth, mean_auc, mean_aucpr, std_auc, std_aucpr, train_time, test_time, path=args.output)


def run_prenet_unseenanomaly(args):
    #    test_list = ['backdoor']
    #    train_list = ['Generic']
    #    test_list = ['analysis']
    #    train_list = ['backdoor', 'Fuzzers', 'backdoor_Fuzzers']
    #    test_list = ['backdoor', 'Generic', 'Fuzzers']
    #    train_list = ['backdoor', 'Generic', 'Fuzzers', 'backdoor_Generic', 'Generic_Fuzzers', 'backdoor_Fuzzers']

    test_list = ['backdoor', 'DoS', 'Fuzzers', 'Reconnaissance']
    train_list = ['backdoor', 'DoS', 'Fuzzers', 'Reconnaissance', \
                  'Reconnaissance_DoS_Fuzzers', 'DoS_backdoor_Fuzzers', 'Reconnaissance_backdoor_Fuzzers', \
                  'Reconnaissance_backdoor_DoS', 'Reconnaissance_DoS', 'Reconnaissance_Fuzzers', \
                  'Reconnaissance_backdoor', 'DoS_backdoor', 'DoS_Fuzzers', 'backdoor_Fuzzers']
    network_depth = int(args.network_depth)
    for nm in test_list:
        for nm2 in train_list:
            if (nm == nm2) or (nm in nm2):
                continue
            filename = 'UNSW_NB15_traintest_' + nm2
            runs = args.runs
            rauc = np.zeros(runs)
            ap = np.zeros(runs)
            global data_format
            data_format = int(args.data_format)
            if data_format == 0:
                x, labels = dataLoading(args.input_path + filename + ".csv")
            else:
                x, labels = get_data_from_svmlight_file(args.input_path + filename + ".svm")
                x = x.tocsr()
            outlier_indices = np.where(labels == 1)[0]
            outliers = x[outlier_indices]
            n_outliers_org = outliers.shape[0]

            train_time = 0
            test_time = 0
            for i in np.arange(runs):
                x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42,
                                                                    stratify=labels)
                y_train = np.array(y_train)
                y_test = np.array(y_test)
                print(filename + ': round ' + str(i))
                outlier_indices = np.where(y_train == 1)[0]
                n_outliers = len(outlier_indices)
                print("Original training size: %d, No. outliers: %d" % (x_train.shape[0], n_outliers))

                n_noise = len(np.where(labels == 0)[0]) * args.cont_rate / (1. - args.cont_rate)
                n_noise = int(n_noise)

                rng = np.random.RandomState(42)
                if data_format == 0:
                    if n_outliers > args.known_outliers:
                        mn = n_outliers - args.known_outliers
                        remove_idx = rng.choice(outlier_indices, mn, replace=False)
                        x_train = np.delete(x_train, remove_idx, axis=0)
                        y_train = np.delete(y_train, remove_idx, axis=0)

                    noises = inject_noise(outliers, n_noise)
                    x_train = np.append(x_train, noises, axis=0)
                    y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))

                else:
                    if n_outliers > args.known_outliers:
                        mn = n_outliers - args.known_outliers
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
                # print(Y)
                input_shape = x_train.shape[1:]
                epochs = args.epochs
                batch_size = args.batch_size
                nb_batch = args.nb_batch

                if network_depth == 2:
                    model = reg_network(input_shape)
                elif network_depth == 1:
                    model = reg_network_no_feature_learner(input_shape)
                else:
                    model = reg_network_deeper(input_shape)

                start_time = time.time()
                model_name = "./model/prenet_" + filename + "_" + str(args.cont_rate) + "cr_" + str(
                    args.batch_size) + "bs_" + str(args.known_outliers) + "ko_" + str(network_depth) + "d.h5"
                checkpointer = ModelCheckpoint(model_name, monitor='loss', verbose=0,
                                               save_best_only=True, save_weights_only=True)
                history = model.fit_generator(
                    pair_generator(x_train, outlier_indices, inlier_indices, Y, batch_size, nb_batch, rng),
                    steps_per_epoch=nb_batch,
                    epochs=epochs,
                    callbacks=[checkpointer])
                train_time += time.time() - start_time

                start_time = time.time()

                print(x_test.shape)
                outlier_indices = np.where(y_test == 1)[0]
                inlier_indices_train = np.where(y_train == 0)[0]
                outlier_indices_train = np.where(y_train == 1)[0]
                #            print(outlier_indices.shape)
                x_test = np.delete(x_test, outlier_indices, axis=0)
                y_test = np.delete(y_test, outlier_indices, axis=0)
                new_anomalies = dataLoading_noheader(args.input_path + nm + '_anomalies_only.csv')
                x_test = np.append(new_anomalies, x_test, axis=0)
                y_test = np.append(np.ones((new_anomalies.shape[0], 1)), y_test)
                scores = load_model_weight_predict(model_name, input_shape, network_depth,
                                                   x_test, x_train[inlier_indices_train],
                                                   x_train[outlier_indices_train])
                rauc[i], ap[i] = aucPerformance(scores, y_test)

            test_name = nm2 + '->>' + nm
            print(test_name)
            mean_auc = np.mean(rauc)
            std_auc = np.std(rauc)
            mean_aucpr = np.mean(ap)
            std_aucpr = np.std(ap)
            train_time = train_time / runs
            test_time = test_time / runs
            print("average AUC-ROC: %.4f, average AUC-PR: %.4f" % (mean_auc, mean_aucpr))
            print("average runtime: %.4f seconds" % (train_time + test_time))
            writeResults(test_name + '_' + str(network_depth), x.shape[0], x.shape[1], n_samples_trn, n_outliers_org,
                         n_outliers,
                         network_depth, mean_auc, mean_aucpr, std_auc, std_aucpr, train_time, test_time,
                         path=args.output)


parser = argparse.ArgumentParser()
parser.add_argument("--network_depth", choices=['1', '2', '4'], default='2',
                    help="the depth of the network architecture")
parser.add_argument("--batch_size", type=int, default=512, help="batch size used in SGD")
parser.add_argument("--nb_batch", type=int, default=20, help="the number of batches per epoch")
parser.add_argument("--epochs", type=int, default=50, help="the number of epochs")
parser.add_argument("--runs", type=int, default=10,
                    help="how many times we repeat the experiments to obtain the average performance")
parser.add_argument("--known_outliers", type=int, default=60, help="the number of labeled outliers available at hand")
parser.add_argument("--cont_rate", type=float, default=0.02, help="the outlier contamination rate in the training data")
parser.add_argument("--ensemble_size", type=int, default=1,
                    help="ensemble_size. Using a size of one runs much faster while being able to obtain similarly good performance as using a size of 30.")
parser.add_argument("--h_lambda", type=float, default=0.01, help="regularization parameter")
parser.add_argument("--ordinal_labels", type=str, default="0,4,8", help="regularization parameter")
parser.add_argument("--input_path", type=str, default='data/', help="the path of the data sets")
# parser.add_argument("--data_set", type=str, default='KDD2014_donors_10feat_nomissing_normalised, census-income-full-mixed-binarized, \
#                     creditcardfraud_normalised, celeba_baldvsnonbald_normalised, UNSW_NB15_traintest_DoS, UNSW_NB15_traintest_Reconnaissance,\
#                     UNSW_NB15_traintest_Fuzzers, UNSW_NB15_traintest_Backdoor, w7a-libsvm-nonsparse_normalised,\
#                     bank-additional-full_normalised, annthyroid_21feat_normalised\
#                     ', help="a list of data set names")
parser.add_argument("--data_set", type=str, default='KDD2014_donors_10feat_nomissing_normalised',
                    help="a list of data set names")
parser.add_argument("--data_format", choices=['0', '1'], default='0',
                    help="specify whether the input data is a csv (0) or libsvm (1) data format")
parser.add_argument("--output", type=str,
                    default='./results/prenet_0.02contrate_2depth_10runs_lambda' + str(h_lambda) + '.csv',
                    help="the output file path")

args = parser.parse_args()

# run_prenet_unseenanomaly(args)
run_prenet(args)

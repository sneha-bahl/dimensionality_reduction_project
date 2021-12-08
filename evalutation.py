import numpy as np
import random
import os, sys
import requests

import urllib
import gzip

from method import dimensionality_reduction
from utils import load_dataset
from autoencoders import Model

# matplotlib
import matplotlib.pyplot as plt

def get_reconstruction_error_curve(method='autoencoder', dataset='mnist'):
    # plot reconstruction error vs hidden size
    num_epochs = 10
    error_log = []
    hidden_size_list = [4, 8, 16, 64, 128, 256]

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset)

    for hidden_size in hidden_size_list:
        # compute reconstruction loss on test data
        error = dimensionality_reduction(X_test, reduced_dimension=hidden_size, X_train=X_train,
                                               get_reconstruction_error=True)
        error_log.append(error)

    return error_log


def get_compression_efficiency(method='autoencoder', dataset='mnist',thresh_acc = 0.75):

    # load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset)

    # set search threshold
    hidden_up = 128
    hidden_low = 4
    acc_up = 1
    acc_low = 0
    # initialize search variables
    hidden_size = hidden_size_prev = 4
    hidden_thresh = 4

    while hidden_up - hidden_low > hidden_thresh:
        hidden_size = int((hidden_up + hidden_low) / 2)
        model = Model(input_size=28 * 28, num_hidden=hidden_size, num_classes=10, classification=False)
        prefix = 'base_model'
        results = model.train(X_train, y_train, X_val, y_val, num_epochs=num_epochs, batch_size=64, lr=1e-2, reg=1e-4,
                              momentum=0, decay=0.3, decay_points=[30, 50, 80], log_interval=4e2, verbose=False)
        # compute reconstruction loss on test data
        X_test_pred = model.predict(X_test)
        error_log.append(np.mean(np.sum((X_test - X_test_pred) ** 2, axis=-1)))

        # extract reduced dimension data
        X_reduced_train = model.get_hidden_features(X_train)
        X_reduced_val = model.get_hidden_features(X_val)
        X_reduced_test = model.get_hidden_features(X_test)

        test_acc = get_classification_error(X_reduced_train, y_train, X_reduced_val, y_val, X_reduced_test, y_test)
        print (hidden_up, hidden_low, hidden_size, test_acc)

        if test_acc < thresh_acc:
            hidden_low = int((hidden_up + hidden_low) / 2)
        else:
            hidden_up = int((hidden_up + hidden_low) / 2)

    hidden_size = int((hidden_up + hidden_low) / 2)
    model = Model(input_size=28 * 28, num_hidden=hidden_size, num_classes=10, classification=False)
    prefix = 'base_model'
    results = model.train(X_train, y_train, X_val, y_val, num_epochs=num_epochs, batch_size=64, lr=1e-2, reg=1e-4,
                          momentum=0, decay=0.3, decay_points=[30, 50, 80], log_interval=4e2, verbose=False)
    # compute reconstruction loss on test data
    X_test_pred = model.predict(X_test)
    error_log.append(np.mean(np.sum((X_test - X_test_pred) ** 2, axis=-1)))

    # extract reduced dimension data
    X_reduced_train = model.get_hidden_features(X_train)
    X_reduced_val = model.get_hidden_features(X_val)
    X_reduced_test = model.get_hidden_features(X_test)

    test_acc = get_classification_error(X_reduced_train, y_train, X_reduced_val, y_val, X_reduced_test, y_test)
    print (
        "Final hidden size: {}\t Test accuracy: {:.2f}%\t Threshold accuracy: {:.2f}%".format(hidden_size,
                                                                                              100 * test_acc,
                                                                                              100 * thresh_acc))
    input_size = X_test.shape[1]
    print ("Compression efficiency: {:.2f}%".format(100 - 100 * hidden_size / input_size))

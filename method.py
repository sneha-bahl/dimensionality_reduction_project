import numpy as np
import random
import os, sys
import requests

import urllib
import gzip

from autoencoders import Model


def dimensionality_reduction(X_in, reduced_dimension=32, X_train=None, method='autoencoder',
                             get_reconstruction_error=False):
    """
        Reduces data dimensionality using the provided method
    """
    assert method in ['autoencoder', 'pca', 'tsne']

    if method == 'autoencoder':
        # normalize data before feeding to neural network (assuming X is not normalized before)
        X_train = (X_train - 0.5) / 0.5
        X_in = (X_in - 0.5) / 0.5

        # train using full training data
        model = Model(input_size=X_train.shape[1], num_hidden=reduced_dimension, classification=False)
        prefix = 'base_model'
        print ("Training autoencoder for reduced dimension: {}".format(reduced_dimension))
        results = model.train(X_train, y_train=np.ones(X_train.shape[0], dtype=np.uint8), X_val=None, y_val=None,
                              num_epochs=10, batch_size=64, lr=1e-2, reg=1e-4,
                              momentum=0, decay=0.3, decay_points=[30, 50, 80], log_interval=4e2, verbose=False)

        print ("Getting reduced dimension data")

        if get_reconstruction_error:
            # get reconstructed data
            X_reconstructed = model.predict(X_in)
            error = np.linalg.norm(X_in-X_reconstructed)
            return error
        else:
            # get reduced dimension data
            X_reduced = model.get_hidden_features(X_in)
            return X_reduced

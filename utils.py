import numpy as np
import os, sys
import requests

import urllib
import gzip
import platform
import pickle

# matplotlib
import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def download_data(dataset_name='fashion-mnist',data_dir='./data/fashion-mnist'):
    if dataset_name == 'fashion-mnist':
        data_dir = os.path.join(data_dir,dataset_name)
        # create data dir if it does not exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        train_data_link = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz"
        train_label_link = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz"
        test_data_link = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz"
        test_label_link = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"

        for url in [train_data_link, train_label_link, test_data_link, test_label_link]:
            basename = os.path.basename(url)
            if not os.path.exists(data_dir + basename):
                r = requests.get(url)
                open(data_dir + basename, 'wb').write(r.content)



# load data function from https://github.com/zalandoresearch/fashion-mnist
def load_mnist_data(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

def load_cifar_batches(path='data/cifar10', kind='train'):
    X_list = []
    y_list = []
    if kind == 'train':
        for i in range(1,6):
            data = unpickle('{}/cifar-10-batches-py/data_batch_{}'.format(path,i))
            X, y = data[b'data'],data[b'labels']
            X_list.append(X)
            y_list.append(y)
        images = np.concatenate(X_list,axis=0)
        labels = np.concatenate(y_list)
    else:
        data = unpickle('{}/cifar-10-batches-py/test_batch'.format(path))
        images,labels = np.array(data[b'data']),np.array(data[b'labels'])
    return images, labels

def load_cifar_data(file):
    """
    This function will read cifar dataset and return data and labels as arrays
    """
    with open(file, 'rb') as fo:
        version = platform.python_version_tuple()
        if version[0] == '2':
            dictionary = pickle.load(fo)
        elif version[0] == '3':
            dictionary = pickle.load(fo, encoding='latin1')
    data = dictionary['data']
    labels = dictionary['labels']
    return data, labels

def load_dataset(dataset_name='fashion-mnist'):
    assert dataset_name in ['fashion-mnist','mnist','cifar-10']

    if dataset_name == 'fashion-mnist' or dataset_name == 'mnist':
        X_train, y_train = load_mnist_data('data/{}'.format(dataset_name), kind='train')
        X_test, y_test = load_mnist_data('data/{}'.format(dataset_name), kind='t10k')

    else:
        X_train, y_train = load_cifar_batches(path='data/cifar10', kind='train')
        X_test, y_test = load_cifar_batches(path='data/cifar10', kind='test')

    # cast X_train, X_test to float
    X_train, X_test = X_train.astype(float) / 255.0, X_test.astype(float) / 255.0

    NUM_VAL = 5000
    NUM_DEV = 500

    idxs = np.arange(len(y_train))
    np.random.shuffle(idxs)
    train_idxs, val_idxs, dev_idxs = idxs[NUM_VAL:], idxs[:NUM_VAL], idxs[NUM_VAL:NUM_VAL + NUM_DEV]

    X_dev, y_dev = X_train[dev_idxs], y_train[dev_idxs]
    X_val, y_val = X_train[val_idxs], y_train[val_idxs]
    X_train, y_train = X_train[train_idxs], y_train[train_idxs]

    print ("Loaded data ... Get data stats:")
    print ("X_train: {} \ty_train: {}".format(X_train.shape, y_train.shape))
    print ("X_val: {} \ty_val: {}".format(X_val.shape, y_val.shape))
    print ("X_test: {} \ty_train: {}".format(X_test.shape, y_test.shape))
    print ("X_dev: {} \ty_dev: {}".format(X_dev.shape, y_dev.shape))

    return X_train, y_train, X_val, y_val, X_test, y_test

def normlize(X_train,X_val,X_test):
    # normalize data
    # compute mean for the training data
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)

    # subtract mean from the data
    # X_train -= X_mean
    # X_val -= X_mean
    # X_test -= X_mean
    # X_dev -= X_mean

    # # scale the data from -1 to 1
    X_train = (X_train - 0.5) / 0.5
    X_val = (X_val - 0.5) / 0.5
    X_test = (X_test - 0.5) / 0.5
    return X_train, X_val, X_test


def visualize(X, title=''):
    mean, std = 0.5, 0.5
    X = std * X + mean
    X = X.reshape(-1, 28, 28)
    X = X[:16]

    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(5, 6))
    for i in range(4):
        for j in range(4):
            ax = axs[i, j]
            ax.imshow(X[4 * i + j], cmap='gray')
            ax.axis(False)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
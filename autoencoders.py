import numpy as np
import random
import os,sys
import requests

import urllib
import gzip

class Dataloader:
    def __init__(self, X, y, batch_size):
        # total_train = int(batch_size * (X.shape[0] // batch_size))
        self.X = X  # [:total_train]
        self.y = y  # [:total_train]
        self.batch_size = batch_size

    def __iter__(self):
        num_iter_per_epoch = len(self.X) // self.batch_size
        idxs = np.arange(len(self.y))
        np.random.shuffle(idxs)
        for i in range(num_iter_per_epoch):
            batch_mask = idxs[i * self.batch_size:(i + 1) * self.batch_size]
            data, labels = self.X[batch_mask], self.y[batch_mask]
            yield data, labels

    def __len__(self):
        return len(self.X) // self.batch_size


# xavier initialization of weights
def xavier(m,h):
    return np.sqrt(6./(m+h))*np.random.uniform(-1,1,size=(m,h))

# kaiming initialization of weights
def kaiming(m,h):
    return np.sqrt(2./m)*np.random.randn(m,h)

def xavier_v2(m,h):
    return np.sqrt(1./m)*np.random.uniform(-1,1,size=(m,h))


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.
    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.
    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)
    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = x.reshape(x.shape[0], -1).dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None

    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = np.sum(dout, axis=0)

    return dx, dw, db


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_forward(x):
    out = sigmoid(x)
    cache = out
    return out, cache


def sigmoid_backward(dout, cache):
    h = cache
    return dout * h * (1 - h)


def tanh_forward(x):
    out = (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)
    cache = out
    return out, cache


def tanh_backward(dout, cache):
    h = cache
    return dout * (1 + h) * (1 - h)


class Model:
    def __init__(self, input_size=28 * 28, num_hidden=400, num_classes=10, classification=False):
        self.input_size = input_size
        self.num_hidden = num_hidden
        self.classification = classification
        self.num_out = num_classes
        self.reset()

    def reset(self):
        # initialize weights using Xavier init
        self.initialize_weights()
        # velocity for the momentum update rule
        self.v_w1, self.v_w2, self.v_b1, self.v_b2 = 0, 0, 0, 0

    def initialize_weights(self):
        # Xavier initialization
        if self.classification:
            out_size = self.num_out
        else:
            out_size = self.input_size

        self.W1 = xavier_v2(self.input_size, self.num_hidden)
        self.W2 = xavier_v2(self.num_hidden, out_size)
        self.b1 = np.sqrt(1. / self.input_size) * np.random.uniform(-1, 1, size=(self.num_hidden))
        self.b2 = np.sqrt(1. / self.num_hidden) * np.random.uniform(-1, 1, size=(out_size))

    def softmax_loss(self, X, y, reg=0):
        # forward propagation
        # (X --> h (-> h_) --> o (-> o_))
        h = X @ self.W1 + self.b1
        h_ = sigmoid(h)
        out = h_ @ self.W2 + self.b2
        softmax_out = np.exp(out) / np.sum(np.exp(out), axis=1).reshape(-1, 1)

        # compute loss
        loss = -np.log(softmax_out[range(out.shape[0]), y]).mean() + 0.5 * reg * (np.sum(self.W1 ** 2)
                                                                                  + np.sum(self.W2 ** 2))

        # back propagation
        dout = softmax_out
        dout[range(out.shape[0]), y] -= 1

        dW2 = h_.T @ dout / X.shape[0] + reg * self.W2
        db2 = np.mean(dout, axis=0)

        dh_ = dout @ self.W2.T
        dh = h_ * (1 - h_) * dh_

        dW1 = X.T @ dh / X.shape[0] + reg * self.W1
        db1 = np.mean(dh, axis=0)
        return loss, dW1, db1, dW2, db2

    def mse_loss(self, X, y, reg=0):
        # forward propagation
        # (X --> h (-> h_) --> o (-> o_))

        h, cache_h = affine_forward(X, self.W1, self.b1)
        h_, cache_h_ = sigmoid_forward(h)

        o, cache_o = affine_forward(h_, self.W2, self.b2)
        o_, cache_o_ = tanh_forward(o)

        # compute loss
        loss = np.mean(np.sum((X - o_) ** 2, axis=-1))

        # backward pass
        dout = 2 * (o_ - X) / X.shape[0]
        dout = tanh_backward(dout, cache_o_)
        dout, dW2, db2 = affine_backward(dout, cache_o)

        dout = sigmoid_backward(dout, cache_h_)
        dout, dW1, db1 = affine_backward(dout, cache_h)

        return loss, dW1, db1, dW2, db2

    # stochastic gradient descent
    def sgd(self, grads, lr, momentum):
        dW1, db1, dW2, db2 = grads
        # Momentum update
        # update velocity
        self.v_w1 = momentum * self.v_w1 - lr * dW1
        self.v_w2 = momentum * self.v_w2 - lr * dW2
        self.v_b1 = momentum * self.v_b1 - lr * db1
        self.v_b2 = momentum * self.v_b2 - lr * db2

        # update position
        self.W1 += self.v_w1
        self.W2 += self.v_w2
        self.b1 += self.v_b1
        self.b2 += self.v_b2

    def train(self, X_train, y_train, X_val, y_val, num_epochs=1, batch_size=64, lr=1e-3, reg=0, classification=False,
              momentum=0, decay=0.1, decay_points=[20, 50, 80], log_interval=200, verbose=True):
        results = []

        for epoch in range(num_epochs):
            train_dataloader = Dataloader(X_train, y_train, batch_size)
            for batch_idx, (data, labels) in enumerate(train_dataloader):
                if not self.classification:
                    loss, dW1, db1, dW2, db2 = self.mse_loss(data, labels, reg)
                else:
                    loss, dW1, db1, dW2, db2 = self.softmax_loss(data, labels, reg)

                # SGD on model paramters
                self.sgd((dW1, db1, dW2, db2), lr, momentum)

                if batch_idx % log_interval == 0 and verbose:
                    print ("Epoch: [{}/{}] \tBatch: [{}/{}] \tloss: {:.4f}".format(epoch + 1, num_epochs,
                                                                                   batch_idx, len(train_dataloader),
                                                                                   loss))
            # decay learning rate
            if epoch in decay_points:
                lr *= decay
                print ("\nreducing learning rate to {:.4f} ...".format(lr))

            # compute train, val accuracy if performing classification
            if self.classification:
                val_loss, val_acc = self.test(X_val, y_val, batch_size=100, reg=reg)
                train_loss, train_acc = self.test(X_train, y_train, batch_size=100, reg=reg)
                results.append([train_loss, val_loss, train_acc, val_acc])
                if verbose:
                    print("\ntrain_loss: {:.4f}\t val_loss: {:.4f}\t train_acc: {:.4f}\t val_acc: {:.4f}\n"
                          .format(train_loss, val_loss, train_acc, val_acc))
        return np.array(results)

    def test(self, X, y, batch_size, reg=0):
        loss, num_correct = 0, 0
        test_dataloader = Dataloader(X, y, batch_size)
        for batch_idx, (data, labels) in enumerate(test_dataloader):
            loss += self.softmax_loss(data, labels, reg)[0] * batch_size
            pred = self.predict(data)
            num_correct += np.sum(pred == labels)
        acc = num_correct / (batch_size * len(test_dataloader))
        loss /= (batch_size * len(test_dataloader))
        return loss, acc

    def predict(self, X):
        # return reconstructed data
        if self.classification:
            h = X @ self.W1 + self.b1
            h_ = sigmoid(h)
            out = h_ @ self.W2 + self.b2
            pred = np.argmax(out, axis=1)
            return pred
        # return classification predictions
        else:
            h, cache_h = affine_forward(X, self.W1, self.b1)
            h_, cache_h_ = sigmoid_forward(h)

            o, cache_o = affine_forward(h_, self.W2, self.b2)
            o_, cache_o_ = tanh_forward(o)
            return o_

    def get_hidden_features(self, X):
        if self.classification:
            h = X @ self.W1 + self.b1
            h_ = sigmoid(h)
            return h_
        else:
            h, cache_h = affine_forward(X, self.W1, self.b1)
            h_, cache_h_ = sigmoid_forward(h)
            return h_
__author__ = 'simon'
import os

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from matplotlib import pyplot

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import cPickle as pickle


def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """

    details = pickle.load(file("features.pkl"))
    X = np.array(details['features'])
    y = np.array(details['ctypes'])

    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.int32)
    else:
        y = None

    return X, y, details['cmap']

if __name__=="__main__":
    X, y, cmap = load()

    z = set(y)
    num_output = len(z)
    print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
        X.shape, X.min(), X.max()))
    print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
        y.shape, y.min(), y.max()))
    net1 = NeuralNet(
        layers=[  # three layers: one hidden layer
            ('input', layers.InputLayer),
            ('dropout1', layers.DropoutLayer),
            ('hidden', layers.DenseLayer),
            #('dropout2', layers.DropoutLayer),
            #('hidden2', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        # layer parameters:
        input_shape=(None, 4096),  # 96x96 input pixels per batch
        hidden_num_units=100,  # number of units in hidden layer
        hidden_nonlinearity=lasagne.nonlinearities.rectify,
        #hidden2_num_units=50,  # number of units in hidden layer
        #hidden2_nonlinearity=lasagne.nonlinearities.rectify,
        output_nonlinearity=lasagne.nonlinearities.softmax,#,  # output layer uses identity function
        output_num_units=len(z),  # 30 target values
        #dropout1_p=0.1,
        #dropout2_p=0.1,
        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.001,
        update_momentum=0.3,

        regression=False,  # flag to indicate we're dealing with regression problem
        max_epochs=1000,  # we want to train this many epochs
        verbose=1,
        )

    net1.fit(X, y)

    with open('net1.pickle', 'wb') as f:
        pickle.dump(net1, f, -1)

    net1.save_params_to("net1_params.pkl")

    train_loss = np.array([i["train_loss"] for i in net1.train_history_])
    valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])
    pyplot.plot(train_loss, linewidth=3, label="train")
    pyplot.plot(valid_loss, linewidth=3, label="valid")
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    #pyplot.ylim(1e-3, 1e-2)
    pyplot.yscale("log")
    pyplot.show()
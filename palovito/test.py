__author__ = 'simon'
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from matplotlib import pyplot

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import cPickle as pickle
from train import load

if __name__=='__main__':
    X,y,cmap = load()

    print len(cmap), type(cmap)
    for k,v in cmap.iteritems():
        print k,v

    net1 = pickle.load(file('net1.pickle','rb'))
    print net1.get_all_params()
    pvs = net1.get_all_params_values()
    for k,v in pvs.iteritems():
        print k,v

    net1.save_params_to("net1_params.pkl")


    y_pred = net1.predict(X)

    print y[-10:]
    print y_pred[-10:]

    crt=0
    for i in range(len(y)):
        if y[i]==y_pred[i]:
            crt+=1

    print "Got %d correct of %d :  %f" % (crt, len(y), float(crt)/float(len(y)))
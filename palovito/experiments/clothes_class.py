__author__ = 'simon'
import os

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from matplotlib import pyplot

import lasagne
import theano
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
import cPickle as pickle
import json
import cv2

def simpleProcessImage(img, amax):
    sz = img.shape

    if sz[0]>sz[1]: # y>x
        w=amax
        h=sz[0]*w/sz[1]
        mid = h/2
        rimg = cv2.resize(img, (w,h))
        crop = rimg[mid-w/2:mid-w/2+amax,:]
        #print 'y>x', w, h, crop.shape
        #cv2.imshow('X',crop)
        #cv2.waitKey(50)
    else:
        h=amax
        w = sz[1]*h/sz[0]
        mid = w/2
        rimg = cv2.resize(img, (w,h))
        crop = rimg[:,mid-h/2:mid-h/2+amax]
        #print 'y<x', w, h, crop.shape

    return crop


def load2d(amax):
    catlists = json.load(file('../catimagelist.json'))
    if False:
        try:
            f = open("images.pkl")
            X,y = pickle.load(f)
            return X,y
        except:
            pass

    features = None
    ctype = []
    cmap = {}

    ct = 0
    imgs = []
    cc = []
    total=0
    for k,v in catlists.iteritems():
        print v['name']
        cnt=0
        cmap[ct] = {'id':k, 'name':v['name']}
        for fn in v['images']:
            img = cv2.imread(fn)
            if img is None:
                continue

            crop = np.array(simpleProcessImage(img, amax)/255.0, dtype=np.float32)
            csz = crop.shape
            #print cmap[ct]
            cv2.imshow('X', crop)
            cv2.waitKey(5)
            if csz[0]==amax and csz[1]==amax:
                imgs.append(crop.transpose(2,0,1))
                cc.append(ct)
                cnt+=1
                total+=1
                if cnt%100==0:
                    print total
            else:
                print 'what?'
            if cnt>=500:
                break
        ct+=1
    if False:
        pickle.dump((np.vstack(imgs).reshape(-1,3,amax,amax), np.array(cc, dtype=np.int32)), open('images.pkl','wb'))
    X = np.vstack(imgs).reshape(-1,3,amax,amax)
    y = np.array(cc, dtype=np.int32)
    print X.shape,y.shape
    #verifyClasses(X, y, cmap)
    #exit()
    X, y = shuffle(X, y, random_state=42)  # shuffle train data
    return X, y, cmap

def getNet2(shape, num_out):
    net2 = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, shape[0],shape[1]),
        conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
        conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
        conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
        hidden4_num_units=500, hidden5_num_units=500,
        output_num_units=num_out, output_nonlinearity=lasagne.nonlinearities.softmax,

        update_learning_rate=0.001,
        update_momentum=0.9,

        regression=False,
        max_epochs=1000,
        verbose=1,
        )
    return net2

class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        return Xb, yb

def float32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

def getNet3(shape, num_out):
    net3 = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),  # !
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('dropout2', layers.DropoutLayer),  # !
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('dropout3', layers.DropoutLayer),  # !
            ('hidden4', layers.DenseLayer),
            ('dropout4', layers.DropoutLayer),  # !
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, shape[0],shape[1]),
        conv1_num_filters=32, conv1_filter_size=(5, 5), pool1_pool_size=(2, 2), dropout1_p=0.1,
        conv2_num_filters=64, conv2_filter_size=(3, 3), pool2_pool_size=(2, 2), dropout2_p=0.2,
        conv3_num_filters=128, conv3_filter_size=(3, 3), pool3_pool_size=(2, 2), dropout3_p=0.5,
        hidden4_num_units=500, dropout4_p=0.5, hidden5_num_units=250,
        output_num_units=num_out, output_nonlinearity=lasagne.nonlinearities.softmax,

        update_learning_rate=theano.shared(float32(0.03)),
        update_momentum=theano.shared(float32(0.9)),

        batch_iterator_train=FlipBatchIterator(batch_size=128),
        on_epoch_finished=[
            AdjustVariable('update_learning_rate', start=0.01, stop=0.0001),
            AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],        regression=False,
        max_epochs=1000,
        verbose=1,
        )
    return net3

def getNet4(shape, num_out):
    net3 = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),  # !
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('dropout2', layers.DropoutLayer),  # !
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('dropout3', layers.DropoutLayer),  # !
            ('hidden4', layers.DenseLayer),
            ('dropout4', layers.DropoutLayer),  # !
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, shape[0],shape[1]),
        conv1_num_filters=32, conv1_filter_size=(5, 5), pool1_pool_size=(2, 2), dropout1_p=0.1,
        conv2_num_filters=64, conv2_filter_size=(3, 3), pool2_pool_size=(2, 2), dropout2_p=0.2,
        conv3_num_filters=128, conv3_filter_size=(3, 3), pool3_pool_size=(2, 2), dropout3_p=0.5,
        hidden4_num_units=250, dropout4_p=0.5, hidden5_num_units=128,
        output_num_units=num_out, output_nonlinearity=lasagne.nonlinearities.softmax,

        update_learning_rate=theano.shared(float32(0.03)),
        update_momentum=theano.shared(float32(0.9)),

        batch_iterator_train=FlipBatchIterator(batch_size=128),
        on_epoch_finished=[
            AdjustVariable('update_learning_rate', start=0.01, stop=0.0001),
            AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],        regression=False,
        max_epochs=1000,
        verbose=1,
        )
    return net3

def verifyClasses(X, y, cmap):
    for i, c in enumerate(y[:100]):
        print i, cmap[c]
        timg = X[i]
        print timg.shape
        img = timg.transpose(1,2,0)
        cv2.imshow("X", img)
        cv2.waitKey()

if __name__=="__main__":
    amax = 96
    if True:
        X, y, cmap = load2d(amax)  # load 2-d data
        print X.shape, y.shape
        if False:
            verifyClasses(X, y, cmap)
            exit()
        net = getNet4((amax,amax),18)
        net.fit(X, y)

        # Training for 1000 epochs will take a while.  We'll pickle the
        # trained model so that we can load it back later:
        import cPickle as pickle
        with open('net2.pickle', 'wb') as f:
            pickle.dump(net, f, -1)
    else:
        net = pickle.load(open('net2.pickle'))

    train_loss = np.array([i["train_loss"] for i in net.train_history_])
    valid_loss = np.array([i["valid_loss"] for i in net.train_history_])
    pyplot.plot(train_loss, linewidth=3, label="train")
    pyplot.plot(valid_loss, linewidth=3, label="valid")
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    #pyplot.ylim(1e-3, 1e-2)
    pyplot.yscale("log")
    pyplot.show()
__author__ = 'simon'
import numpy as np
import theano
import theano.tensor as T
from lafeat import fetch_overfeat_weights_and_biases
from sklearn_theano.feature_extraction.overfeat_class_labels import get_overfeat_class_label

import lasagne

class Standardize(lasagne.layers.Layer):
    def __init__(self, incoming, mean, std, **kwargs):
        super(Standardize, self).__init__(incoming, **kwargs)
        self.mean = mean
        self.std = std

    def get_output_for(self, input, **kwargs):
        return (input - self.mean)/self.std

    def get_output_shape_for(self, input_shape):
        return input_shape

def buildOverFeat(input_var, W, b):
    network = lasagne.layers.InputLayer(shape=(None, 3, 231,231),
                                    input_var=input_var)

    network = Standardize(network, 118.380948, 61.896913)

    network = lasagne.layers.Conv2DLayer(network, 96, (11,11), W=W[0], b=b[0])
    network = lasagne.layers.MaxPool2DLayer(network, (2,2))

    network = lasagne.layers.Conv2DLayer(network, 256, (5,5), W=W[1], b=b[1])
    network = lasagne.layers.MaxPool2DLayer(network, (2,2))

    network = lasagne.layers.Conv2DLayer(network, 512, (3,3), W=W[2], b=b[2])
    network = lasagne.layers.Conv2DLayer(network, 1024, (3,3), W=W[3], b=b[3])
    network = lasagne.layers.Conv2DLayer(network, 1024, (3,3), W=W[4], b=b[4])
    network = lasagne.layers.MaxPool2DLayer(network, (2,2))

    network = lasagne.layers.Conv2DLayer(network, 3072, (6,6), W=W[5], b=b[5])
    network = lasagne.layers.Conv2DLayer(network, 4096, (1,1), W=W[6], b=b[6])
    network = lasagne.layers.Conv2DLayer(network, 1000, (1,1), nonlinearity=None, W=W[7], b=b[7])
    #network = lasagne.layers.DenseLayer(network, 1000, W=W[7], b=b[7])
    return network

def buildSimpleNet(input_var, W, b):
    network = lasagne.layers.InputLayer(shape=(None, 3, 231,231),
                                    input_var=input_var)

    network = Standardize(network, 118.380948, 61.896913)
    #network = lasagne.layers.TransformerLayer()
    network = lasagne.layers.Conv2DLayer(network, 96, (11,11), stride=(4,4), W=W[0], b=b[0])
    network = lasagne.layers.MaxPool2DLayer(network, (2,2))

    network = lasagne.layers.Conv2DLayer(network, 256, (5,5), W=W[1], b=b[1])
    network = lasagne.layers.MaxPool2DLayer(network, (2,2))


    network = lasagne.layers.Conv2DLayer(network, 512, (3,3), pad=1, W=W[2], b=b[2])

    network = lasagne.layers.Conv2DLayer(network, 1024, (3,3), pad=1, W=W[3], b=b[3])
    network = lasagne.layers.Conv2DLayer(network, 1024, (3,3), pad=1, W=W[4], b=b[4])
    network = lasagne.layers.MaxPool2DLayer(network, (2,2))
    print W[5].shape, b[5].shape
    """
    network = lasagne.layers.Conv2DLayer(network, 3072, (6,6), W=W[5], b=b[5])

    print W[6].shape
    network = lasagne.layers.DenseLayer(network, 4096, W=W[6].transpose(2,3,1,0)[0,0,:,:], b=b[6])
    network = lasagne.layers.DenseLayer(network, 1000, nonlinearity=None, W=W[7].transpose(2,3,1,0)[0,0,:,:], b=b[7])
    #network = lasagne.layers.Conv2DLayer(network, 4096, (1,1), W=W[6], b=b[6])
    #network = lasagne.layers.Conv2DLayer(network, 1000, (1,1), nonlinearity=None, W=W[7], b=b[7])
    """
    return network

def buildSimpleNetNoWeight(input_var, W, b):
    network = lasagne.layers.InputLayer(shape=(None, 3, 231,231),
                                    input_var=input_var)

    network = Standardize(network, 118.380948, 61.896913)
    #network = lasagne.layers.TransformerLayer()
    network = lasagne.layers.Conv2DLayer(network, 96, (11,11), stride=(4,4))
    network = lasagne.layers.MaxPool2DLayer(network, (2,2), stride=(2,2))

    network = lasagne.layers.Conv2DLayer(network, 256, (5,5))
    network = lasagne.layers.MaxPool2DLayer(network, (2,2), stride=(2,2))


    network = lasagne.layers.Conv2DLayer(network, 512, (3,3), pad=1)

    network = lasagne.layers.Conv2DLayer(network, 1024, (3,3), pad=1)
    network = lasagne.layers.Conv2DLayer(network, 1024, (3,3), pad=1)
    network = lasagne.layers.MaxPool2DLayer(network, (2,2), stride=(2,2))

    network = lasagne.layers.Conv2DLayer(network, 3072, (6,6))

    network = lasagne.layers.DenseLayer(network, 4096)
    network = lasagne.layers.DenseLayer(network, 1000, nonlinearity=None)
    #network = lasagne.layers.Conv2DLayer(network, 4096, (1,1), W=W[6], b=b[6])
    #network = lasagne.layers.Conv2DLayer(network, 1000, (1,1), nonlinearity=None, W=W[7], b=b[7])

    return network

if __name__=="__main__":
    import cv2
    import pylab

    from training_images import simpleProcessImage
    img = cv2.imread("/home/simon/python/sklearn-theano/sklearn_theano/datasets/images/sloth_closeup.jpg")
    #img = np.asarray((img-118.380948)/61.896913, np.float32)
    crop = simpleProcessImage(img)
    input_var = T.tensor4('inputs')
    W, b = fetch_overfeat_weights_and_biases()

    if False:
        for i, w in enumerate(W):
            print i, w.shape

        overfeat = buildOverFeatNoWeight(input_var, W, b)
        prediction = lasagne.layers.get_output(overfeat, deterministic=True)
        pred_fn = theano.function([input_var], prediction)
    else:
        simple = buildSimpleNet(input_var, W, b)
        prediction = lasagne.layers.get_output(simple)
        pred_fn = theano.function([input_var], prediction)


    recrop = crop.transpose(2,0,1).reshape(-1,3,231,231)
    print recrop.shape
    res = pred_fn(recrop)
    print res.shape
    exit()
    print res.max(axis=1)
    exp_res = np.exp(res - res.max(axis=1))
    exp_res /= np.sum(exp_res, axis=1)
    indices = np.argsort(res, axis=1)
    indices = indices[:, -5:]
    class_strings = np.empty_like(indices,
                                  dtype=object)
    for index, value in enumerate(indices.flat):
        class_strings.flat[index] = get_overfeat_class_label(value)
        print exp_res[0,value], class_strings.flat[index]
    print class_strings

    pylab.subplot(1, 3, 1); pylab.axis('off'); pylab.imshow(crop)
    pylab.show()
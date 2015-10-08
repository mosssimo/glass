__author__ = 'simon'
import os
import numpy as np
import theano
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
import index_net as inet
from sklearn_theano.datasets import get_dataset_dir

# better get it from a config file
NETWORK_WEIGHTS_PATH = get_dataset_dir("overfeat_weights")
SMALL_NETWORK_FILTER_SHAPES = np.array([(96, 3, 11, 11),
                                        (256, 96, 5, 5),
                                        (512, 256, 3, 3),
                                        (1024, 512, 3, 3),
                                        (1024, 1024, 3, 3),
                                        (3072, 1024, 6, 6),
                                        (4096, 3072, 1, 1),
                                        (1000, 4096, 1, 1)])

SMALL_NETWORK_WEIGHT_FILE = 'net_weight_0'
SMALL_NETWORK_BIAS_SHAPES = SMALL_NETWORK_FILTER_SHAPES[:, 0]
SMALL_NETWORK = (SMALL_NETWORK_WEIGHT_FILE,
                 SMALL_NETWORK_FILTER_SHAPES,
                 SMALL_NETWORK_BIAS_SHAPES)
"""
            Convolution(ws[0], bs[0], subsample=(4, 4),
                        activation='relu'),
            MaxPool((2, 2)),

            Convolution(ws[1], bs[1], activation='relu'),
            MaxPool((2, 2)),

            Convolution(ws[2], bs[2],
                        activation='relu',
                        cropping=[(1, -1), (1, -1)],
                        border_mode='full'),

            Convolution(ws[3], bs[3],
                        activation='relu',
                        cropping=[(1, -1), (1, -1)],
                        border_mode='full'),

            Convolution(ws[4], bs[4],
                        activation='relu',
                        cropping=[(1, -1), (1, -1)],
                        border_mode='full'),
            MaxPool((2, 2)),

            Convolution(ws[5], bs[5],
                        activation='relu'),

            Convolution(ws[6], bs[6],
                        activation='relu'),

            Convolution(ws[7], bs[7],
                        activation='identity')]

"""
def fetch_overfeat_weights_and_biases(large_network=False, weights_file=None):
    network =  SMALL_NETWORK
    fname, weight_shapes, bias_shapes = network

    if weights_file is None:
        weights_file = os.path.join(NETWORK_WEIGHTS_PATH, fname)
        """
        if not os.path.exists(weights_file):
            url = "https://dl.dropboxusercontent.com/u/15378192/net_weights.zip"
            if not os.path.exists(NETWORK_WEIGHTS_PATH):
                os.makedirs(NETWORK_WEIGHTS_PATH)
            full_path = os.path.join(NETWORK_WEIGHTS_PATH, "net_weights.zip")
            if not os.path.exists(full_path):
                download(url, full_path, progress_update_percentage=1)
            zip_obj = zipfile.ZipFile(full_path, 'r')
            zip_obj.extractall(NETWORK_WEIGHTS_PATH)
            zip_obj.close()
        """
    memmap = np.memmap(weights_file, dtype=np.float32)
    mempointer = 0

    weights = []
    biases = []
    for weight_shape, bias_shape in zip(weight_shapes, bias_shapes):
        filter_size = np.prod(weight_shape)
        weights.append(
            memmap[mempointer:mempointer + filter_size].reshape(weight_shape))
        mempointer += filter_size
        biases.append(memmap[mempointer:mempointer + bias_shape])
        mempointer += bias_shape

    return weights, biases
if __name__=="__main__":
    W, b = fetch_overfeat_weights_and_biases()

    net1 = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('conv2d1', layers.Conv2DLayer),
                ('maxpool1', layers.MaxPool2DLayer),
                ('conv2d2', layers.Conv2DLayer),
                ('maxpool2', layers.MaxPool2DLayer),
                ('conv2d3', layers.Conv2DLayer),
                ('conv2d4', layers.Conv2DLayer),
                ('conv2d5', layers.Conv2DLayer),
                ('maxpool3', layers.MaxPool2DLayer),
                ('conv2d6', layers.Conv2DLayer),
                ('conv2d7', layers.Conv2DLayer),
                ('conv2d8', layers.Conv2DLayer)
                #('output', layers.DenseLayer)
               ],
        # input layer
        input_shape=(None, 3, 231, 231),
        # layer conv2d1
        conv2d1_num_filters=96,
        conv2d1_filter_size=(11, 11),
        conv2d1_stride=(4,4),
        conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d1_W=W[0],
        # layer maxpool1
        maxpool1_pool_size=(2, 2),
        # layer conv2d2
        conv2d2_num_filters=256,
        conv2d2_filter_size=(5, 5),
        conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d2_W=W[1],
        maxpool2_pool_size=(2, 2),

        # layer conv2d3
        conv2d3_num_filters=512,
        conv2d3_filter_size=(3,3),
        conv2d3_pad=1,
        conv2d3_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d3_W=W[2],

         # layer conv2d4
        conv2d4_num_filters=1024,
        conv2d4_filter_size=(3,3),
        conv2d4_pad=1,
        conv2d4_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d4_W=W[3],

        # layer conv2d5
        conv2d5_num_filters=1024,
        conv2d5_filter_size=(3,3),
        conv2d5_pad=1,
        conv2d5_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d5_W=W[4],

        maxpool3_pool_size=(2, 2),

        # layer conv2d6
        conv2d6_num_filters=3072,
        conv2d6_filter_size=(6,6),
        conv2d6_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d6_W=W[5],

        # layer conv2d7
        conv2d7_num_filters=4096,
        conv2d7_filter_size=(1,1),
        conv2d7_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d7_W=W[6],

        # layer conv2d8
        conv2d8_num_filters=1000,
        conv2d8_filter_size=(1,1),
        conv2d8_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d8_W=W[7],

        #output_nonlinearity=lasagne.nonlinearities.softmax,#,  # output layer uses identity function
        #output_num_units=1000,  # 1000 target values
        #output_W = W[7],

        # optimization method params
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,
        max_epochs=10,
        verbose=1,
        regression=True
    )
    for i, w in enumerate(W):
        print i, w.shape

    net1.initialize()
    import cv2
    from training_images import simpleProcessImage
    img = cv2.imread("/home/simon/python/sklearn-theano/sklearn_theano/datasets/images/cat_and_dog.jpg")

    crop = simpleProcessImage(img)
    cv2.imshow("X", crop)
    res = net1.predict(crop.transpose(2,0,1).reshape(-1,3,231,231))
    print res

    cv2.waitKey()

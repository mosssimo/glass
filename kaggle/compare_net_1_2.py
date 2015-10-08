__author__ = 'simon'
import numpy as np

from net1 import load
from net2 import load2d
from matplotlib import pyplot

from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    if y is not None:
        axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

def detect(img, cascade_fn='/home/simon/Opencv/opencv-3.0.0/data/haarcascades/haarcascade_frontalface_alt.xml',
           scaleFactor=1.3, minNeighbors=4, minSize=(20, 20)):

    cascade = cv2.CascadeClassifier(cascade_fn)
    rects = cascade.detectMultiScale(img, scaleFactor=scaleFactor,
                                     minNeighbors=minNeighbors,
                                     minSize=minSize)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


if __name__=="__main__":
    import cPickle as pickle
    import cv2

    img = cv2.imread("/home/simon/Downloads/george-clooney.jpg")
    img = cv2.imread("/home/simon/Pictures/smoss.jpg")
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detect(grey)
    print rects, len(rects)
    face = rects[0]
    grey = grey[face[1]:face[3],face[0]:face[2]]/255.0
    grey = cv2.resize(grey, (96,96))
    draw_rects(img, rects, (0,255,0))
    cv2.imshow("X", grey)

    net1 = pickle.load(file('net1.pickle'))
    net2 = pickle.load(file('net2.pickle'))

    sample1 = load(test=True)[0][6:7]
    print sample1


    sample2 = load2d(test=True)[0][6:7]
    print grey.shape
    sample2 = grey.reshape(-1,1,96,96)
    sample2 = np.array(sample2, np.float32)
    print sample2

    y_pred1 = net1.predict(sample1)[0]
    import time

    for i in range(100):
        y_pred2 = net2.predict(sample2)[0]
        print i

    fig = pyplot.figure(figsize=(6, 3))
    ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
    plot_sample(sample1[0], y_pred1, ax)
    ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
    plot_sample(sample2[0], y_pred2, ax)
    pyplot.show()

    train_loss = np.array([i["train_loss"] for i in net1.train_history_])
    valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])
    pyplot.plot(train_loss, linewidth=3, label="train")
    pyplot.plot(valid_loss, linewidth=3, label="valid")
    train_loss2 = np.array([i["train_loss"] for i in net2.train_history_])
    valid_loss2 = np.array([i["valid_loss"] for i in net2.train_history_])
    pyplot.plot(train_loss2, linewidth=3, label="train")
    pyplot.plot(valid_loss2, linewidth=3, label="valid")
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    pyplot.ylim(1e-3, 1e-2)
    pyplot.yscale("log")
    pyplot.show()
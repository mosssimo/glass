__author__ = 'simon'
import json
import numpy as np
import cv2
from sklearn_theano.feature_extraction import OverfeatTransformer
import cPickle as pickle

def simpleProcessImage(img):
    sz = img.shape

    if sz[0]>sz[1]: # y>x
        w=231
        h=sz[0]*w/sz[1]
        mid = h/2
        rimg = cv2.resize(img, (w,h))
        crop = rimg[mid-w/2:mid+w/2+1,:]
        print 'y>x', crop.shape
    else:
        h=231
        w = sz[1]*h/sz[0]
        mid = w/2
        rimg = cv2.resize(img, (w,h))
        crop = rimg[:,mid-h/2:mid+h/2+1]
        print 'y<x', crop.shape

    return crop

if __name__=='__main__':
    catlists = json.load(file('catimagelist.json'))
    #print catlists
    tranformer = OverfeatTransformer(output_layers=[-2])

    features = None
    ctype = []
    cmap = {}

    ct = 0
    for k,v in catlists.iteritems():
        print v['name']
        cnt=0
        cmap[ct] = {'id':k, 'name':v['name']}
        for fn in v['images']:
            img = cv2.imread(fn)
            if img is None:
                continue

            crop = simpleProcessImage(img)
            csz = crop.shape
            if csz[0]==231 and csz[1]==231:
                res = tranformer.transform(crop)
            else:
                continue
            #print res.shape
            #print res[res>0.0]

            if features is None:
                features = res
            else:
                features = np.vstack((features, res))
            ctype.append(ct)
            #cv2.imshow('X', crop)
            #cv2.waitKey()
            cnt+=1
            if cnt>500:
                break
        ct += 1
    print 'final size', features.shape
    details = {'cmap':cmap, 'ctypes':ctype, 'features':features}
    pickle.dump(details, file("features.pkl", 'wb'))
__author__ = 'simon'
'''
Created on Sep 24, 2012

@author: simon
'''
import boto3
from boto3.session import Session

import json
import numpy as np

def unify(obj):
    #print obj
    if isinstance(obj, str):
        return unicode(obj, 'latin')
    elif isinstance(obj, dict):
        nd = {}
        #print obj
        for k, v in obj.iteritems():
            #print "unifying ", k
            nk = unify(k)
            #print "--> ", nk
            #print "unifying ", v
            nv = unify(v)
            #print "--> ", nv
            nd[nk] = nv
        return nd
    elif isinstance(obj, list):
        nl = []
        for item in obj:
            nl.append(unify(item))
        return nl
    elif isinstance(obj, tuple):
        nl = []
        for item in obj:
            nl.append(unify(item))
        return tuple(nl)
    else:
        return obj

def denumpy(npObj):
    if type(npObj)==dict:
        o = {}
        for k,v in npObj.iteritems():
            vp = denumpy(v)
            o[k] = vp
        return o
    elif type(npObj)==list:
        o = []
        for v in npObj:
            o.append(denumpy(v))
        return o
    elif 'tolist' in dir(npObj):
        return npObj.tolist()
    else:
        return npObj

if __name__="__main__":
    s3 = boto3.resource('s3')
    key = ""
    obj = s3.Object(bucket_name='palla', key=key)
    print(obj.bucket_name)
    print(obj.key)
    response = obj.get()
    data = response['Body'].read()
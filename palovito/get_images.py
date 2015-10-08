
__author__ = 'simon'
from pymongo import MongoClient
import json
import requests
import time

settings = {
    "mongo_db":"ec2-52-18-225-198.eu-west-1.compute.amazonaws.com",
    "mongo_port":27017,
}

def fetchAndStoreImage(url, asin):
    try:
        res = requests.get(url)
        f = open("/home/simon/data/palovito/images/%s.jpg" % asin, "wb")
        f.write(res.content)
        print "image %d saved." % cnt
        f.close()
        return True

    except Exception, e:
        print "Exception :", e
        print "sleeping a little longer.."
        return False

def fetchAndStoreProductImage(p):
    if 'PrimaryImage' in p:
        if p['PrimaryImage']!=[]:
            success = fetchAndStoreImage(p['PrimaryImage'], p['ASIN'])
            if success:
                time.sleep(1)
            else:
                # download error, maybe being throttled, pause a little longer
                time.sleep(5)
            return success
        else:
            #print json.dumps(p, indent=2)
            return False
    else:
        print "PrimaryImage not set in :", p
    return False

class MongoStore(object):
    def __init__(self, clientDB):
        self.mongo = MongoClient(settings['mongo_db'], settings['mongo_port'])
        self.mongodb = self.mongo[clientDB]

    def getAllProducts(self):
        products = []
        print self.mongodb.amazon_products.count()
        cnt = 0

        for p in self.mongodb.amazon_products.find().skip(2434+2426).batch_size(50):
            print p['ASIN']
            fetchAndStoreProductImage(p)
            cnt+=1

if __name__=="__main__":
    mongoStore = MongoStore("UK")
    import cv2

    if False:
        mongoStore.getAllProducts()
    else:
        from sql import PGConnector
        from groups import getTopCategories

        pgcs = PGConnector({"host":"quackdb.cozdd0etxhab.eu-west-1.rds.amazonaws.com", "user":"mick", "password":"crewe2wba0", "dbname":"uk"})
        cats = getTopCategories(pgcs)
        catlists = {}
        fetch = False

        for c in cats:
            if c[2]>1000:
                cnt=0
                cat = {'name':c[0], 'id':c[1], 'images':[]}
                print c
                for p in mongoStore.mongodb.amazon_products.find({'BrowseNodes.0.BrowseNodeId':c[1]}).batch_size(100):
                    del p['_id']
                    #print json.dumps(p, indent=2)
                    try:
                        fn = "/home/simon/data/palovito/images/%s.jpg" %p['ASIN']
                        #print fn
                        f = open(fn)
                        f.close()
                        if cnt<200 and False:
                            img = cv2.imread(fn)
                            cv2.imshow('X', img)
                            cv2.waitKey(50)
                        cat['images'].append(fn)
                        cnt+=1
                    except IOError, e:
                        #print type(e),dir(e)
                        #print "image doesnt exist"
                        if fetch:
                            fetchAndStoreProductImage(p)
                    except Exception, e:
                        print e
                        exit()
                print cnt
                if cnt>=500:
                    catlists[c[1]] = cat
                #cv2.waitKey()

        for k in catlists:
            print k, catlists[k]['name']
        json.dump(catlists, file('catimagelist.json','wb'))
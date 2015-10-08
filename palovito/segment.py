__author__ = 'simon'
import cv2
from matplotlib import pyplot as plt
import numpy as np

def getBackgroundMask(img):
    img2 = img.copy()
    #img2[:11,:] = img2[0,0]
    mask = np.zeros(img.shape[:2],dtype = np.uint8) # mask initialized to PR_BG

    sz = img.shape
    rect = (20,10,sz[1]-20,sz[0]-10)
    bgdmodel = np.zeros((1,65),np.float64)
    fgdmodel = np.zeros((1,65),np.float64)
    cv2.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==1) + (mask==3), 255, 0).astype('uint8')
    return mask2

if __name__=="__main__":
    img = cv2.imread("/home/simon/data/palovito/images/B00B1A301E.jpg")

    mask = getBackgroundMask(img)
    cv2.imshow("M", mask)
    cv2.waitKey()
    #mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
    #output = cv2.bitwise_and(img2,img2,mask=mask2)

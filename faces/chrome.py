__author__ = 'simon'
import cv2

if __name__=='__main__':
    img = cv2.imread('/home/simon/Pictures/smoss.jpg')
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("X", grey)
    print "hrllo"
    cv2.waitKey()
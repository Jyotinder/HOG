#!/usr/local/bin/python2.7
import cv2
import os
from Vlad import *
from scipy.cluster.vq import *
def sift(image_path):
    """
        Input:  image_path
                path to raw image (./image/airplane/airplane1.tif)
        Output: fd
                SIFT FV
        Note:: To configure HOG parameter use config file to set
                orientations, pixels_per_cell, cells_per_block, visualize, normalize
                size of fd depend upoin these paprameter
    """
    print "##################"
    print "SIFT Enter"+image_path
    im = cv2.imread(image_path)
    rows,cols,ch = im.shape
    if rows!= 256 or cols !=256:
        print image_path
        print rows, cols
        im = cv2.resize(im,(256, 256), interpolation = cv2.INTER_CUBIC)
        return []
    # Create feature extraction and keypoint detector objects
    fea_det = cv2.FeatureDetector_create("SIFT")
    des_ext = cv2.DescriptorExtractor_create("SIFT")
    kpts = fea_det.detect(im)
    kpts, des = des_ext.compute(im, kpts)
    print "##################"
    print "SIFT End"
    return des

def shiftToVlad(des):
    k=32
    voc, variance = kmeans(des, k, 1)
    vladVector= vlad(des,voc)
    if k*128!=vladVector.size:
        return None
    return vladVector









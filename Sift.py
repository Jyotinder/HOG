#!/usr/local/bin/python2.7
from skimage.feature import hog
from config import *
from sklearn.cross_validation import train_test_split
import argparse as ap
import cv2
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import matplotlib.pyplot as plt

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def HOG(image_path):

    """
        Input:  image_path
                path to raw image (./image/airplane/airplane1.tif)
        Output: fd
                HOG FV
        Note:: To configure HOG parameter use config file to set
                orientations, pixels_per_cell, cells_per_block, visualize, normalize
                size of fd depend upoin these paprameter
    """
    print image_path
    im = cv2.imread(image_path)
    rows,cols,ch = im.shape
    if rows!= 256 or cols !=256:
        print image_path
        print rows, cols
        im = cv2.resize(im,(256, 256), interpolation = cv2.INTER_CUBIC)
        #return []
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    split= blockshaped(im, 256/8, 256/8)
    fd=[]
    for image in split:
        fd.append(  hog(image, orientations, pixels_per_cell, cells_per_block, visualize, normalize))
    return fd
def SIFT_old(image_path):

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

def SIFT(image_path):

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
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    split= blockshaped(im, 256/4, 256/4)
    descriptors = np.array([], dtype=np.float).reshape(0,128)
    #fd=[]
    for image in split:
        kpts = fea_det.detect(image)
        kpts, des = des_ext.compute(image, kpts)
        if des !=[] and des is not None :
            descriptors=np.vstack([descriptors,des])
            #fd.append(des)
    return descriptors

def imlist(path):
    """
    The function imlist returns all the names of the files in
    the directory path supplied as argument to the function.
    """
    return [os.path.join(path, f) for f in os.listdir(path)]

def find_nn(point, neighborhood):
    """
    Finds the nearest neighborhood of a vector.

    Args:
        point (float array): The initial point.
        neighborhood (numpy float matrix): The points that are around the initial point.

    Returns:
        float array: The point that is the nearest neighbor of the initial point.
        integer: Index of the nearest neighbor inside the neighborhood list
    """
    min_dist = float('inf')
    nn = neighborhood[0]
    nn_idx = 0
    for i in range(len(neighborhood)):
        neighbor = neighborhood[i]
        dist = cv2.norm(point - neighbor)
        if dist < min_dist:
            min_dist = dist
            nn = neighbor
            nn_idx = i

    return nn, nn_idx

def vlad(descriptors, centers):
    """
    Calculate the Vector of Locally Aggregated Descriptors (VLAD) which is a global descriptor from a group of
    descriptors and centers that are codewords of a codebook, obtained for example with K-Means.

    Args:
        descriptors (numpy float matrix): The local descriptors.
        centers (numpy float matrix): The centers are points representatives of the classes.

    Returns:
        numpy float array: The VLAD vector.
    """
    dimensions = len(descriptors[0])
    vlad_vector = np.zeros((len(centers), dimensions), dtype=np.float)
    for descriptor in descriptors:
        nearest_center, center_idx = find_nn(descriptor, centers)
        for i in range(dimensions):
            vlad_vector[center_idx][i] += (descriptor[i] - nearest_center[i])
    # L2 Normalization
    vlad_vector = cv2.normalize(vlad_vector)
    vlad_vector = vlad_vector.flatten()
    return vlad_vector


def sift(train_path):
    training_names = os.listdir(train_path)
    # Get all the path to the images and save them in a list
    # image_paths and the corresponding label in image_paths
    image_paths = []
    image_classes = []
    class_id = 0
    for training_name in training_names:#Got three directory having images
        dir = os.path.join(train_path, training_name)
        class_path = imlist(dir)
        image_paths+=class_path
        image_classes+=[class_id]*len(class_path)
        class_id+=1

    ################################################
    #############SPLIT DATA#########################
    X_train, X_test, y_train, y_test = train_test_split(image_paths, image_classes, test_size=0.2,random_state=0)
    ################################################
    # List where all the descriptors are stored
    des_list = []
    descriptors = np.array([], dtype=np.float).reshape(0,128)
    k=64
    x=[]
    for i,image_path in enumerate(X_train):
        des=SIFT_old(image_path)
        if des !=[] and des is not None :
            descriptors=np.vstack([descriptors,des])
            k_means = KMeans(init='k-means++', n_clusters=k, n_init=5)
            k_means.fit(descriptors)
            voc = k_means.cluster_centers_
            vladVector =vlad(descriptors,voc)
            print vladVector.size
            if vladVector.size !=k*128:
                print "Error with 128"
            x.append(vladVector)
            des_list.append((image_path,y_train[i],des))
        else:
            del X_train[i]
            del y_train[i]

    clf = LinearSVC()
    print "SVM Start"
    y_train=[]
    for i in des_list:
        y_train.append(i[1])
    print len(x),"  ",len(y_train)
    clf.fit(x, y_train)
    print "SVM End"
    #####################TEST######################
    # List where all the descriptors are stored
    x_test=[]
    ytest=[]
    for i,image_path in enumerate(X_test):
        des=SIFT_old(image_path)
        if des !=[] and des is not None :
            k_means = KMeans(init='k-means++', n_clusters=k, n_init=5)
            k_means.fit(descriptors)
            voc = k_means.cluster_centers_
            vladVector =vlad(descriptors,voc)
            print vladVector.size
            if vladVector.size !=k*128:
                print "Error with 128"
            x_test.append(vladVector)
            ytest.append(y_test[i])
        else:
            del X_test[i]
            del y_test[i]

    predictions =  clf.predict(x_test)
    print(classification_report(ytest, predictions, target_names=training_names))
    #print predictions

    print "Confusion S"
    cm = confusion_matrix(ytest, predictions)
    plt.figure()
    plot_confusion_matrix(cm)
    plt.show()
    print "Confusion E"

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, Cname=""):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(Cname))
    plt.xticks(tick_marks, Cname, rotation=45)
    plt.yticks(tick_marks, Cname)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



if __name__ == '__main__':
    #hog_fuc("./Images")
    sift("./Images")

#!/usr/local/bin/python2.7
from sklearn.cross_validation import train_test_split
import argparse as ap
import cv2
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


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
    kpts = fea_det.detect(im)
    kpts, des = des_ext.compute(im, kpts)
    print "##################"
    print "SIFT End"

    return des

def imlist(path):
    """
    The function imlist returns all the names of the files in
    the directory path supplied as argument to the function.
    """
    return [os.path.join(path, f) for f in os.listdir(path)]

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
    for image_path in X_train:
        des=SIFT(image_path)
        if des !=[] and des is not None :
            descriptors=np.vstack([descriptors,des])
            des_list.append((image_path,des))
        else:
            print "IMP"+image_path

    k = 100
    print "K mean"
    voc, variance = kmeans(descriptors, k, 1)
    print "END K mean"
    # Calculate the histogram of features
    im_features = np.zeros((len(X_train), k), "float32")
    for i in xrange(len(des_list)):
        words, distance = vq(des_list[i][1],voc)
        for w in words:
            im_features[i][w] += 1


    # Scaling the words
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)

    # Train the Linear SVM
    clf = LinearSVC()
    clf.fit(im_features, y_train)
    # Save the SVM
    print "SAV SVM"
    joblib.dump((clf, training_names, stdSlr, k, voc), "bof_new.pkl", compress=3)

    #####################TEST######################
    # List where all the descriptors are stored
    des_list = []
    descriptors = np.array([], dtype=np.float).reshape(0,128)
    for image_path in X_test:
        des=SIFT(image_path)
        if des !=[] and des is not None :
            print "Add into Stack"
            descriptors=np.vstack([descriptors,des])
            des_list.append((image_path,des))
            print "Added"
        else:
            print "IMP"+image_path

    # Calculate the histogram of features
    im_features = np.zeros((len(X_test), k), "float32")
    for i in xrange(len(des_list)):
        words, distance = vq(des_list[i][1],voc)
        for w in words:
            im_features[i][w] += 1

    # Scaling the words
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)

    predictions =  clf.predict(im_features)
    print(classification_report(y_test, predictions, target_names=training_names))
    #print predictions

    print "Confusion S"
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    print "Confusion E"



def test(path_svm,path_image):
    clf, classes_names, stdSlr, k, voc = joblib.load(path_svm)
    # Create feature extraction and keypoint detector objects
    fea_det = cv2.FeatureDetector_create("SIFT")
    des_ext = cv2.DescriptorExtractor_create("SIFT")
    # List where all the descriptors are stored
    descriptors = np.array([], dtype=np.float).reshape(0,128)
    des=SIFT(path_image)
    if des !=[] and des is not None :
        descriptors=np.vstack([descriptors,des])
    else:
        print "IMP"+path_image

    test_features = np.zeros((1, k), "float32")
    words, distance = vq(descriptors,voc)
    for w in words:
        test_features[0][w] += 1

    # Scale the features
    test_features = stdSlr.transform(test_features)

    predictions =  clf.predict(test_features)
    print predictions



if __name__ == '__main__':
    sift("./Images")
    # if os.path.isfile("bofj.pkl"):
    #     test("bofj.pkl","Images/mobilehomepark/mobilehomepark05.tif")
    # else:
    #     sift("./Images")

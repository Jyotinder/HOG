from skimage.feature import hog
from config import *
import cv2
import numpy as np
import argparse as ap
import os
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.metrics import classification_report
import csv
from itertools import chain

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
    im = cv2.imread(image_path)
    rows,cols,ch = im.shape
    if rows!= 256 or cols !=256:
        print image_path
        print rows, cols
        im = cv2.resize(im,(256, 256), interpolation = cv2.INTER_CUBIC)
        #return []
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
    return fd

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
    im = cv2.imread(image_path)
    rows,cols,ch = im.shape
    if rows!= 256 or cols !=256:
        print image_path
        print rows, cols
        im = cv2.resize(im,(256, 256), interpolation = cv2.INTER_CUBIC)
        #return []
    # Create feature extraction and keypoint detector objects
    fea_det = cv2.FeatureDetector_create("SIFT")
    des_ext = cv2.DescriptorExtractor_create("SIFT")
    kpts = fea_det.detect(im)
    kpts, des = des_ext.compute(im, kpts)

    # Stack all the descriptors vertically in a numpy array
    descriptors = des
    descriptors = np.vstack((descriptors, des))

    #
    test_features = np.zeros((len(image_paths), k), "float32")
    for i in xrange(len(image_paths)):
        words, distance = vq(des_list[i][1],voc)
        for w in words:
            test_features[i][w] += 1

    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
    idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

    # Scale the features
    test_features = stdSlr.transform(test_features)

    return des

def feature_extraction(folder_path,choice):


    """
        Input:  folder_path
                path to raw image folder(./image) having dirctory structure as
                ./image/argriculture
                ./image/airplane

        Output: data
                Hog Feature vetcor

                y_true
                True lable for each image for exampe all the FV of arigculture will have 0

                training_names
                String name of all the classes Or the Subfolder in the ./image

        Note:: This function call HOG to get FV for image
    """

    data = []
    y_true = []
    count = 0
    training_names = os.listdir(folder_path)
    for dir in training_names:
        dir_file_path = os.path.join(folder_path, dir)
        files = os.listdir(dir_file_path)
        for file in files:
            vector=[]
            if choice:
                vector = HOG(os.path.join(dir_file_path,file))
            else:
                vector = SIFT(os.path.join(dir_file_path,file))
            #if vector != []:
            data.append(vector)
            y_true.append(count)
        count = count + 1
    return data, y_true, training_names


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


def cv_estimate(n_folds=5,X=[],y=[],name=[]):
    """
        Input:  X array of FV
                Y true class of each vector
                n_folds number of fold you want to create
        Output: Generate Confusion matrix

        Note:: This function use KFold which on each n_folds iteration
                gives train set and test set on the X data set which are random and mutually
                exclusive. Each train set work as a min batch(incremental set) on which SGD is trained
                I union test set to create confusion matrix.
    """
    cv = KFold(len(X), n_folds=n_folds)
    #K Fold


    y_pred=[]
    for train, test in cv:
        X_partial_train=[]
        y_partial_train=[]
        for i in train:
            X_partial_train.append(X[i])
            y_partial_train.append(y[i])
        clf = linear_model.SGDClassifier()
        clf.fit(X_partial_train, y_partial_train)
        X_test=[]
        y_test=[]
        for i in test:
            X_test.append(X[i])
            y_test.append(y[i])
        y_temp=clf.predict(X_test)
        print  accuracy_score(y_test, y_temp)
        print(classification_report(y_test, y_temp, target_names=name))


def main():
    parser = ap.ArgumentParser()
    parser.add_argument('-d', "--folderpath", help="Path to Image", required=True)
    args = vars(parser.parse_args())
    path = args["folderpath"]

    name=[]
    data_X=[]
    ture_y=[]
    if HOG_E:
        if os.path.isfile("feat.hog"):

            data_X, ture_y = joblib.load("./feat.hog")
            name=os.listdir(path)
            #result=np.array(list(csv.reader(open("Hog.csv","rb"),delimiter=','))).astype('float')

        else:
            data_X, ture_y, name = feature_extraction(path,1)
            joblib.dump((data_X, ture_y),"feat.hog")
            res = zip(ture_y,data_X)
            with open(r'Hog.csv', 'wb') as fout:
                csvout = csv.writer(fout)
                csvout.writerows(res)


        #cv_estimate(n_folds=5,X=data_X, y=ture_y,name=name)
        for loop in range(0,3):
            print "###############################################"

            clf = linear_model.SGDClassifier()
            X_train, X_test, y_train, y_test = train_test_split(data_X, ture_y, test_size=0.2,random_state=0)
            clf.fit(X_train, y_train)
            y_temp=clf.predict(X_test)
            print  accuracy_score(y_test, y_temp)
            print(classification_report(y_test, y_temp, target_names=name))
            print "###############################################"

            print "Confusion S"
            cm = confusion_matrix(y_test, y_temp)
            print(cm)
            print "Confusion E"
    if SIFT_E:
        print "SFIT Enabled"
        if os.path.isfile("feat.SFIT"):

            data_X, ture_y = joblib.load("./feat.SFIT")
            name=os.listdir(path)
            #result=np.array(list(csv.reader(open("Hog.csv","rb"),delimiter=','))).astype('float')

        else:
            data_X, ture_y, name = feature_extraction(path,0)
            joblib.dump((data_X, ture_y),"feat.SFIT")
            res = zip(ture_y,data_X)
            with open(r'Hog.csv', 'wb') as fout:
                csvout = csv.writer(fout)
                csvout.writerows(res)
        #cv_estimate(n_folds=5,X=data_X, y=ture_y,name=name)
        print "###############################################"

        clf = linear_model.SGDClassifier()
        X_train, X_test, y_train, y_test = train_test_split(data_X, ture_y, test_size=0.2,random_state=0)
        clf.fit(X_train, y_train)
        y_temp=clf.predict(X_test)
        print  accuracy_score(y_test, y_temp)
        print(classification_report(y_test, y_temp, target_names=name))
        print "###############################################"



if __name__ == "__main__":
    main()

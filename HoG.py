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
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.svm import LinearSVC
from sklearn import tree
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


def feature_extraction(folder_path):


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
            vector = HOG(os.path.join(dir_file_path,file))
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


def cv_estimate(n_folds=5,X=[],y=[]):
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
    clf = linear_model.SGDClassifier()
    #K Fold

    y_test=[]
    y_pred=[]
    for train, test in cv:
        X_partial_train=[]
        y_partial_train=[]
        for i in train:
            X_partial_train.append(X[i])
            y_partial_train.append(y[i])
        clf.partial_fit(X_partial_train, y_partial_train,classes=[i for i in range(0,21)])
        X_test=[]
        for i in test:
            X_test.append(X[i])
            y_test.append(y[i])

        y_temp=clf.predict(X_test)
        for j in y_temp:
            y_pred.append(j)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure()
        plot_confusion_matrix(cm)
        plt.show()
        print cm


def main():
    parser = ap.ArgumentParser()
    parser.add_argument('-d', "--folderpath", help="Path to Image", required=True)
    args = vars(parser.parse_args())
    path = args["folderpath"]


    data_X=[]
    ture_y=[]
    if os.path.isfile("feat.hog"):

        data_X, ture_y = joblib.load("./feat.hog")
        #result=np.array(list(csv.reader(open("Hog.csv","rb"),delimiter=','))).astype('float')

    else:
        data_X, ture_y, name = feature_extraction(path)
        joblib.dump((data_X, ture_y),"feat.hog")
        res = zip(ture_y,data_X)
        with open(r'Hog.csv', 'wb') as fout:
            csvout = csv.writer(fout)
            csvout.writerows(res)


    cv_estimate(n_folds=5,X=data_X, y=ture_y)





if __name__ == "__main__":
    main()

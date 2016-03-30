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
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.svm import LinearSVC
from sklearn import tree

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


def main():
    parser = ap.ArgumentParser()
    parser.add_argument('-d', "--folderpath", help="npz folder", required=True)
    args = vars(parser.parse_args())
    path = args["folderpath"]

    data_X, ture_y, name = feature_extraction(path)
    print name
    print ture_y

    clf = 0
    X_train, X_test, y_train, y_test = train_test_split(data_X, ture_y, test_size=0.2, random_state=0)
    if clf_option ==1:
        clf = linear_model.SGDClassifier()
        clf.partial_fit(X_train, y_train)
    elif clf_option ==2:
        clf = LinearSVC(C=1)
        clf.fit(X_train, y_train)
    else:
        clf = tree.DecisionTreeClassifier()
        clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, Cname=name)
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm)
    plt.show()
    print cross_val_score(clf, data_X, ture_y, cv=5, verbose=1)




if __name__ == "__main__":
    main()

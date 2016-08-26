from sklearn.cross_validation import train_test_split
import cv2
import os
import numpy as np

from sklearn.svm import LinearSVC
from scipy.cluster.vq import *
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from Sift import *

def imlist(path):
    """
    The function imlist returns all the names of the files in
    the directory path supplied as argument to the function.
    """
    return [os.path.join(path, f) for f in os.listdir(path)]

def direcrtoryProcessing(train_path):
    training_names = os.listdir(train_path)
    # Get all the path to the images and save them in a list
    # image_paths and the corresponding label in image_paths
    image_paths = []
    image_classes = []
    class_id = 0
    for training_name in training_names:
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
    for i,image_path in enumerate(X_train):
        des=sift(image_path)
        if des !=[] and des is not None :
            vlad=shiftToVlad(des)
            if vlad != None:
                des_list.append((image_path,y_train[i],vlad))
            else:
                del X_train[i]
                del y_train[i]
        else:
            del X_train[i]
            del y_train[i]
    x=[]
    for i in xrange(len(des_list)):
        x.append(des_list[i][2])

    clf = LinearSVC()
    print "SVM Start"
    y_train=[]
    for i in des_list:
        y_train.append(i[1])
    print len(x),"  ",len(y_train)
    clf.fit(x, y_train)
    print "SVM End"
    #####################TEST######################
    #List where all the descriptors are stored
    x_test=[]
    ytest=[]
    for i,image_path in enumerate(X_test):
        des=sift(image_path)
        if des !=[] and des is not None :
            vlad=shiftToVlad(des)
            if vlad != None:
                 x_test.append(vlad)
                 ytest.append(y_test[i])
            else:
                del X_test[i]
                del y_test[i]
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
    direcrtoryProcessing("./Images")
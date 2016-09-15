from sklearn.cross_validation import train_test_split
import cv2
import os
import numpy as np

from sklearn.svm import LinearSVC
#from scipy.cluster.vq import *
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from Sift import *
import itertools
import random
import sklearn.cross_validation.KFold as KFold

def imlist(path):
    """
    The function imlist returns all the names of the files in
    the directory path supplied as argument to the function.
    """
    return [os.path.join(path, f) for f in os.listdir(path)]


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
   clf = SGDClassifier()
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



def trainTestSet(setX,setY):
    x_trainingList=[]
    y_train=[]
    for i,image_path in enumerate(setX):
        # reduce the column of the image to 128 doesn't change the number of rows
        des=PCA_image(image_path)
        if des !=[] and des is not None:
            k = 128
            k_means = KMeans(init='k-means++', n_clusters=k, n_init=10)
            k_means.fit(des)
            voc = k_means.cluster_centers_
            #reduce the size of the row to the value of k
            vlad= vladFun(des,voc)
            x_trainingList.append(vlad)
            y_train.append(setY[i])
        else:
                del setX[i]
                del setY[i]
    return x_trainingList,y_train

def batches(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def direcrtoryProcessing(train_path):
    training_names = os.listdir(train_path)
    class_id = 0
    train_X=[]
    train_Y=[]
    test_X=[]
    test_Y=[]

    for training_name in training_names:
        dir = os.path.join(train_path, training_name)
        class_path = imlist(dir)
        image_paths=class_path
        image_classes=[class_id]*len(class_path)
        XX_train, XX_test, yy_train, yy_test=train_test_split(image_paths, image_classes, test_size=0.2,random_state=0)
        train_X.append(XX_train)
        train_Y.append(yy_train)
        test_X.append(XX_test)
        test_Y.append(yy_test)
        class_id+=1

    # this is done to make it a single list
    train_X = list(itertools.chain.from_iterable(train_X))
    train_Y = list(itertools.chain.from_iterable(train_Y))

    x_trainingList,y_train=trainTestSet(train_X,train_Y)
    print len(x_trainingList),"  ",len(y_train)
    test_X = list(itertools.chain.from_iterable(test_X))
    test_Y = list(itertools.chain.from_iterable(test_Y))
    x_test,ytest=trainTestSet(test_X,test_Y)

    shuffledRange = range(len(x_trainingList))
    sdg=SGDClassifier()
    n_iter = 5
    for n in range(n_iter):
        random.shuffle(shuffledRange)
        shuffledX = [x_trainingList[i] for i in shuffledRange]
        shuffledY = [y_train[i] for i in shuffledRange]
        for batch in batches(range(len(shuffledX)), 10):
            sdg.partial_fit(shuffledX[batch[0]:batch[-1]+1], shuffledY[batch[0]:batch[-1]+1], classes=np.unique(y_train))


        #sdg.partial_fit(x_trainingList,y_train,classes=np.unique(y_train))
    predictions = sdg.predict(x_test)
    print(classification_report(ytest, predictions))



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
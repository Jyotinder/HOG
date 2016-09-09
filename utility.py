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
from Vlad import vladFun
from Sift import *
from sklearn.externals import joblib
from sklearn.decomposition import PCA

def imlist(path):
    """
    The function imlist returns all the names of the files in
    the directory path supplied as argument to the function.
    """
    return [os.path.join(path, f) for f in os.listdir(path)]





def trainTestSet(setX,setY):
    x_trainingList=[]
    y_train=[]
    for i,image_path in enumerate(setX):
        des=PCA_image(image_path)
        if des !=[] and des is not None and des.size/128 >=128:
            k = 64
            k_means = KMeans(init='k-means++', n_clusters=k, n_init=10)
            k_means.fit(des)
            voc = k_means.cluster_centers_
            vlad= vladFun(des,voc)
            x_trainingList.append(vlad)
            y_train.append(setY[i])
        else:
                del setX[i]
                del setY[i]
    return x_trainingList,y_train



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
    x_trainingList,y_train=trainTestSet(X_train,y_train)
    print len(x_trainingList),"  ",len(y_train)
    sdg=SGDClassifier()
    sdg.partial_fit(x_trainingList,y_train,classes=np.unique(y_train))
    #clf = LinearSVC()
    #clf.fit(x_trainingList, y_train)
    print "SVM End"
    #####################TEST######################
    #List where all the descriptors are stored
    x_test,ytest=trainTestSet(X_test,y_test)
    #predictions = clf.predict(x_test)
    predictions = sdg.predict(x_test)
    print(classification_report(ytest, predictions, target_names=training_names))
    #print predictions



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
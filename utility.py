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
from Vlad import vladFun
from Sift import *
from sklearn.externals import joblib

def imlist(path):
    """
    The function imlist returns all the names of the files in
    the directory path supplied as argument to the function.
    """
    return [os.path.join(path, f) for f in os.listdir(path)]


def testSet(setX,setY,class_id):
    des_list = []
    for i,image_path in enumerate(setX):
        des=siftPyramid(image_path)
        if des !=[] and des is not None :
            des_list.append((image_path,setY[i],des))
        else:
                del setX[i]
                del setY[i]
    des_list = sorted(des_list, key=lambda tup: tup[1])
    x_trainingList=[]
    for imageClass in range(0,class_id):
        temp=[x for x in des_list if x[1]==imageClass]
        descriptors = np.array([], dtype=np.float).reshape(0,128)
        for x in temp:
            descriptors=np.vstack([descriptors,x[2]])
        print("Load Kmean for class "+ str(imageClass))
        filename = './Kmean/'+str(imageClass)+".pkl"
        voc=joblib.load(filename)
        print("Vlad for training images for class"+ str(imageClass))
        for i in xrange(len(temp)):
            vlad= vladFun(temp[i][2],voc)
            print(vlad.size)
            x_trainingList.append(vlad)
    y_train=[]
    for i in des_list:
        y_train.append(i[1])
    return x_trainingList,y_train

def trainTestSet(setX,setY,class_id):
    des_list = []
    for i,image_path in enumerate(setX):
        des=siftPyramid(image_path)
        if des !=[] and des is not None :
            des_list.append((image_path,setY[i],des))
        else:
                del setX[i]
                del setY[i]
    des_list = sorted(des_list, key=lambda tup: tup[1])
    x_trainingList=[]
    for imageClass in range(0,class_id):
        temp=[x for x in des_list if x[1]==imageClass]
        descriptors = np.array([], dtype=np.float).reshape(0,128)
        for x in temp:
            descriptors=np.vstack([descriptors,x[2]])
        k = 128
        print("Kmean for class "+ str(imageClass))
        #voc, variance = kmeans(descriptors, k, 1)
        k_means = KMeans(init='k-means++', n_clusters=k, n_init=10)
        k_means.fit(descriptors)
        k_means_labels = k_means.labels_
        voc = k_means.cluster_centers_
        k_means_labels_unique = np.unique(k_means_labels)

        filename = './Kmean/'+str(imageClass)+".pkl"
        joblib.dump(voc, filename, compress=9)
        print("Vlad for training images for class"+ str(imageClass))
        for i in xrange(len(temp)):
            vlad= vladFun(temp[i][2],voc)
            print(vlad.size)
            x_trainingList.append(vlad)
    print "SVM Start"
    y_train=[]
    for i in des_list:
        y_train.append(i[1])
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
    x_trainingList,y_train=trainTestSet(X_train,y_train,class_id)
    print len(x_trainingList),"  ",len(y_train)
    clf = LinearSVC()
    clf.fit(x_trainingList, y_train)
    filename = './Kmean/'+"clf.pkl"
    joblib.dump(clf, filename, compress=9)
    print "SVM End"
    #####################TEST######################
    #List where all the descriptors are stored
    x_test,ytest=testSet(X_test,y_test,class_id)
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
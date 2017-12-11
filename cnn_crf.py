##### This code is developed to extract deep features using cnn, train and test with CRF instead of lstm for temporal info  ####

from __future__ import print_function
from keras.models import Sequential,Model
from keras.layers import LSTM
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, TimeDistributed
from keras.optimizers import SGD, rmsprop
#from keras.utils.data_utils import get_file
#from sklearn.metrics import confusion_matrix
from keras.layers import Flatten, Dense, Dropout, Activation, Reshape, Permute, Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils
#from keras.optimizers import SGD
from convnetskeras.convnets import preprocess_image_batch, convnet
from keras.callbacks import ModelCheckpoint
#import sys
#sys.path.insert(0,'/usr/local/lib/python2.7/dist-packages/keras/')
#import theano.tensor as T
#import pickle
import random
#import glob
#import os
#import scipy
#import scipy.io
#from keras import backend as K
import glob
import csv
from sklearn.metrics import confusion_matrix
np.set_printoptions(threshold=np.nan)
#from seqlearn.perceptron import StructuredPerceptron
#from seqlearn.evaluation import bio_f_score
from sklearn.svm import LinearSVC
#from pystruct.models import ChainCRF
#from pystruct.learners import FrankWolfeSSVM
#########################################################
#### Reading the images
#########################################################
INPUT_LEN = 22
INPUT_DIM = 4096
OUTPUT_LEN = 4

batch_size = 32
nb_classes = 4
data_augmentation = False

# input image dimensions
img_rows, img_cols = 320, 320
# the images are RGB
img_channels = 3
#

data=np.load('BVZ0072_BVZ0073_97pots.npz')
print('hey')

TOT_0=data['TOT_0_seg']
TOT_90=data['TOT_90_seg']
TOT_180=data['TOT_180_seg']
TOT_270=data['TOT_270_seg']
GL=data['Plant_label']
y1=data['Plant_label_single']
Tray=data['Tray_label_single']
BVZ=data['BVZ_label_single']


TOT_0=np.transpose(TOT_0,(0,3,1,2))
TOT_90=np.transpose(TOT_90,(0,3,1,2))
TOT_180=np.transpose(TOT_180,(0,3,1,2))
TOT_270=np.transpose(TOT_270,(0,3,1,2))

m=[103.939, 116.779, 123.68]
TOT_0_r = np.zeros((TOT_0.shape[0],3,227,227),dtype=np.uint8)
for i in range(TOT_0.shape[0]):
    temp = TOT_0[i,0,:,:]
    temp2 = cv2.resize(temp,(227,227))
    TOT_0_r[i,2,:,:] = temp2-m[0]
    temp = TOT_0[i,1,:,:]
    temp2 = cv2.resize(temp,(227,227))
    TOT_0_r[i,1,:,:] = temp2-m[1]
    temp = TOT_0[i,2,:,:]
    temp2 = cv2.resize(temp,(227,227))
    TOT_0_r[i,0,:,:] = temp2-m[2]
    TOT_0_r[i,:,:,:] = np.expand_dims(TOT_0_r[i,:,:,:], axis=0)
print(TOT_0_r.shape[2])
TOT_90_r = np.zeros((TOT_90.shape[0],3,227,227),dtype=np.uint8)
for i in range(TOT_90.shape[0]):
    temp = TOT_90[i,0,:,:]
    temp2 = cv2.resize(temp,(227,227))
    TOT_90_r[i,2,:,:] = temp2-m[0]
    temp = TOT_90[i,1,:,:]
    temp2 = cv2.resize(temp,(227,227))
    TOT_90_r[i,1,:,:] = temp2-m[1]
    temp = TOT_90[i,2,:,:]
    temp2 = cv2.resize(temp,(227,227))
    TOT_90_r[i,0,:,:] = temp2-m[2]
    TOT_90_r[i,:,:,:] = np.expand_dims(TOT_90_r[i,:,:,:], axis=0)
TOT_180_r = np.zeros((TOT_180.shape[0],3,227,227),dtype=np.uint8)
for i in range(TOT_180.shape[0]):
    temp = TOT_180[i,0,:,:]
    temp2 = cv2.resize(temp,(227,227))
    TOT_180_r[i,2,:,:] = temp2-m[0]
    temp = TOT_180[i,1,:,:]
    temp2 = cv2.resize(temp,(227,227))
    TOT_180_r[i,1,:,:] = temp2-m[1]
    temp = TOT_180[i,2,:,:]
    temp2 = cv2.resize(temp,(227,227))
    TOT_180_r[i,0,:,:] = temp2-m[2]
    TOT_180_r[i,:,:,:] = np.expand_dims(TOT_180_r[i,:,:,:], axis=0)
TOT_270_r = np.zeros((TOT_270.shape[0],3,227,227),dtype=np.uint8)
for i in range(TOT_270.shape[0]):
    temp = TOT_270[i,0,:,:]
    temp2 = cv2.resize(temp,(227,227))
    TOT_270_r[i,2,:,:] = temp2-m[0]
    temp = TOT_270[i,1,:,:]
    temp2 = cv2.resize(temp,(227,227))
    TOT_270_r[i,1,:,:] = temp2-m[1]
    temp = TOT_270[i,2,:,:]
    temp2 = cv2.resize(temp,(227,227))
    TOT_270_r[i,0,:,:] = temp2-m[2]
    TOT_270_r[i,:,:,:] = np.expand_dims(TOT_270_r[i,:,:,:], axis=0)

np.savez('Data_7_97j_seg', TOT_0_r=TOT_0_r, TOT_90_r=TOT_90_r, TOT_180_r=TOT_180_r, TOT_270_r=TOT_270_r, GL=GL, y1=y1, Tray=Tray, BVZ=BVZ)


print(y1)
lab1=np.sum(y1==1)
lab2=np.sum(y1==2)
lab3=np.sum(y1==3)
lab4=np.sum(y1==4)
print(np.sum(y1==1))
print(np.sum(y1==2))
print(np.sum(y1==3))
print(np.sum(y1==4))

#    print(GL)
print(GL.shape[0])
print(TOT_0_r.shape[0])
     
 

#########################################################
#### CNN feature extraction
#########################################################

#CNN_model = convnet('alexnet',weights_path="alexnet_weights.h5", heatmap=False)
CNN_model = convnet('alexnet24',weights_path="./tmp/weights_alex_7_1.hdf5", heatmap=False)
#        CNN_model = convnet('vgg_16',weights_path="vgg16_weights.h5", heatmap=False)


from keras import backend as K
get_dense_features = K.function([CNN_model.layers[0].input, K.learning_phase()],[CNN_model.layers[31].output]) #33

print ("Feature extraction...")
X_0 = np.zeros((97, INPUT_LEN, INPUT_DIM))
X_90 = np.zeros((97, INPUT_LEN, INPUT_DIM))
X_180 = np.zeros((97, INPUT_LEN, INPUT_DIM))
X_270 = np.zeros((97, INPUT_LEN, INPUT_DIM))
y = np.zeros((97, OUTPUT_LEN))

for newcounter in range(0, 97):
    for inp_len in range (0, 22):
       
        current_frame = np.zeros((1,3,227,227))
        current_frame = np.array(np.expand_dims(TOT_0_r[newcounter*22+inp_len,:,:,:], axis=0))
        X_0[newcounter, inp_len] = np.array(get_dense_features([current_frame,0]))[0]
        current_frame = np.zeros((1,3,227,227))
        current_frame = np.array(np.expand_dims(TOT_90_r[newcounter*22+inp_len,:,:,:], axis=0))
        X_90[newcounter, inp_len] = np.array(get_dense_features([current_frame,0]))[0]
        current_frame = np.zeros((1,3,227,227))
        current_frame = np.array(np.expand_dims(TOT_180_r[newcounter*22+inp_len,:,:,:], axis=0))
        X_180[newcounter, inp_len] = np.array(get_dense_features([current_frame,0]))[0]
        current_frame = np.zeros((1,3,227,227))
        current_frame = np.array(np.expand_dims(TOT_270_r[newcounter*22+inp_len,:,:,:], axis=0))
        X_270[newcounter, inp_len] = np.array(get_dense_features([current_frame,0]))[0]

    y[newcounter,GL[newcounter*22+10]-1] = 1
    print (newcounter)
       

INPUT_LEN = 22
INPUT_DIM = 4096
OUTPUT_LEN = 4
ind_tot = range(97)
random.seed(16)
rand00=random.sample(ind_tot,97)


for j in range(5):

    BATCH_SIZE = 32
        
    ind_test_0 = []
    ind_train_0 = []
    ind_val_0 = []


    if j!= 4 :
        ind_test_0 = rand00[j*20:(j+1)*20]
        ind_val_0 = rand00[82:97]
    else :
        ind_test_0 = rand00[77:97]
        ind_val_0 = rand00[0:15]

    ind_train_0 = list(set(ind_tot) - set(ind_test_0) - set(ind_val_0))


    ind_test_90 = []  
    ind_train_90 = []
    ind_val_90 = []
    ind_test_180 = []  
    ind_train_180 = []   
    ind_val_180 = []
    ind_test_270 = []  
    ind_train_270 = []   
    ind_val_270 = []
     
    ind_test_90 = ind_test_0
    ind_val_90 = ind_val_0 
    ind_train_90 = ind_train_0
    ind_test_180 = ind_test_0
    ind_val_180 = ind_val_0 
    ind_train_180 = ind_train_0 
    ind_test_270 = ind_test_0 
    ind_val_270 = ind_val_0


    ind_train_270 = ind_train_0
          
          
    newYtrain = []
    newYtest = []
    newYval = []
    X_train1 = []
    X_val1 = []
    X_test1 = []


    X_test1 = np.concatenate((X_0[ind_test_0,:,:],X_90[ind_test_90,:,:],X_180[ind_test_180,:,:],X_270[ind_test_270,:,:]),axis=0)
    X_val1 = np.concatenate((X_0[ind_val_0,:,:],X_90[ind_val_90,:,:],X_180[ind_val_180,:,:],X_270[ind_val_270,:,:]),axis=0)
    X_train1 = np.concatenate((X_0[ind_train_0,:,:],X_90[ind_train_90,:,:],X_180[ind_train_180,:,:],X_270[ind_train_270,:,:]),axis=0)
    newYtest=np.concatenate((y[ind_test_0],y[ind_test_90],y[ind_test_180],y[ind_test_270]),axis=0)
    newYval=np.concatenate((y[ind_val_0],y[ind_val_90],y[ind_val_180],y[ind_val_270]),axis=0)
    newYtrain=np.concatenate((y[ind_train_0],y[ind_train_90],y[ind_train_180],y[ind_train_270]),axis=0)

    ny= np.array([])
    ny=np.zeros(newYtrain.shape[0])
    for nf in range(0, newYtrain.shape[0]):
        for df in range(4) :
            if newYtrain[nf,df]==1:
                ny[nf]=df
    nyt= np.array([])
    nyt=np.zeros(newYtest.shape[0])
    for nf in range(0, newYtest.shape[0]):
        for df in range(4) :
            if newYtest[nf,df]==1:
                nyt[nf]=df


    np.savez('Data_alexnet', X_train1=X_train1, X_test1=X_test1, newYtrain=newYtrain, newYtest=newYtest)
    exit()


    X_tr=[None]*248
    for ii in range(0,248):
        X_tr[ii]=X_train1[ii]
    y_tr=[None]*248
    for ii in range(0,248):
        y_tr[ii]=np.array([ny[ii],ny[ii],ny[ii],ny[ii],ny[ii],ny[ii],ny[ii],ny[ii],ny[ii],ny[ii],ny[ii],ny[ii],ny[ii],ny[ii],ny[ii],ny[ii],ny[ii],ny[ii],ny[ii],ny[ii],ny[ii],ny[ii]])

    X_tr1=np.asarray(X_tr)
    y_tr1=np.asarray(y_tr)   
    y_tr1 = y_tr1.astype(int)

    # Train linear SVM
#    svm = LinearSVC(dual=False, C=.1)
    # flatten input
#    svm.fit((X_train1), (ny))
#    print("Test score with linear SVM: %f" % svm.score((X_test1), (nyt)))
    # Train linear chain CRF
    model = ChainCRF()
    ssvm = FrankWolfeSSVM(model=model, C=.1, max_iter=200)
    ssvm.fit(X_tr1, y_tr1)


    print('--------------------')
    N_test = X_test1.shape[0]
    len_seq = X_test1.shape[1]
    X_ts=[None]*N_test
    for ii in range(0,N_test):
        X_ts[ii]=X_test1[ii]
    y_ts=[None]*N_test
    for ii in range(0,N_test):
        y_ts[ii]=np.array([nyt[ii]]*len_seq)

    X_ts1=np.asarray(X_ts)
    y_ts1=np.asarray(y_ts)   
    y_ts1 = y_ts1.astype(int)

    out_crf = ssvm.predict(X_ts1)
    labels_test_crf = np.zeros(N_test)
    for ii in range(N_test):
        (unique, counts) = np.unique(out_crf[ii], return_counts=True)
        nums = np.argmax(counts)
        labels_test_crf[ii] = unique[nums]

    # Train linear SVM
    svm = LinearSVC(dual=False, C=.1)
    svm.fit(np.vstack(X_tr1), np.hstack(y_tr1))

    out_s = svm.predict(np.vstack(X_ts1))
    out_svm = [np.array(out_s[ii*len_seq : (ii+1)*len_seq]) for ii in range(N_test)]
    labels_test_svm = np.zeros(N_test)
    for ii in range(N_test):
        print("SVM: ", np.array(out_svm[ii]))
        print("CRF: ", np.array(out_crf[ii]))
        (unique, counts) = np.unique(out_svm[ii], return_counts=True)
        print("G-T: ", nyt[ii])
        nums = np.argmax(counts)
        labels_test_svm[ii] = unique[nums]

    print('')
    print("Test score with linear SVM: %f" % svm.score(np.vstack(X_ts1), np.hstack(y_ts1)))
    print('--------------------')
    print("Test score with chain  CRF: %f" % ssvm.score(X_ts1, y_ts1))
    print('--------------------')
    print("Ground truth    :", nyt)
    print("SVM Predictions :", labels_test_svm)
    print("CRF Predictions :", labels_test_crf)
    print('--------------------')
    acc_svm = float(np.sum(nyt==labels_test_svm))/float(N_test)   
    print("Total score (sequence) with linear SVM: %f" %acc_svm)
    print('--------------------')
    acc_crf = float(np.sum(nyt==labels_test_crf))/float(N_test)   
    print("Total score (sequence) with chain CRF: %f" %acc_crf)
    print('--------------------')

    print('')
    print('')


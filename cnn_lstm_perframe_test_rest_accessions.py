##### This code is developed to classify other accessions   ####

from __future__ import print_function
import csv
from keras.models import Sequential,Model
from keras.layers import LSTM, SimpleRNN
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
import array
np.set_printoptions(threshold=np.nan)

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


TOT_0=np.load('BVZ0072_BVZ0073_rest1.npy')
data2=np.load('BVZ0072_BVZ0073_rest5.npy')

print('hey')


GL=data2[0]
y1=data2[1]
Tray=data2[3]
BVZ=data2[5]
letter=data2[6]

TOT_0=np.transpose(TOT_0,(0,3,1,2))
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


#########################################################
#### CNN feature extraction
#########################################################

#CNN_model = convnet('alexnet',weights_path="alexnet_weights.h5", heatmap=False)
###CNN_model = convnet('alexnet24',weights_path="./tmp/weights_alex_7_1.hdf5", heatmap=False)
CNN_model = convnet('alexnet24',weights_path="./tmp/weights_alex_7_tot.hdf5", heatmap=False)
#        CNN_model = convnet('vgg_16',weights_path="vgg16_weights.h5", heatmap=False)


from keras import backend as K
get_dense_features = K.function([CNN_model.layers[0].input, K.learning_phase()],[CNN_model.layers[31].output]) #33

print ("Feature extraction...")
X_0 = np.zeros((455, INPUT_LEN, INPUT_DIM))
y = np.zeros((455, INPUT_LEN, OUTPUT_LEN))

for newcounter in range(0, 455):
    for inp_len in range (0, 22):
       
        current_frame = np.zeros((1,3,227,227))
        current_frame = np.array(np.expand_dims(TOT_0_r[newcounter*22+inp_len,:,:,:], axis=0))
        X_0[newcounter, inp_len] = np.array(get_dense_features([current_frame,0]))[0]

    print (newcounter)
       
#########################################################
#### Preparing the data for training the LSTM model
#########################################################
######################################################
#### LSTM Model
#########################################################
INPUT_LEN = 22
INPUT_DIM = 4096
OUTPUT_LEN = 4
ind_tot = range(455)
#random.seed(16)
#rand00=random.sample(ind_tot,97)
#print('@@@@@@@@@@@@@@@@@@@@@@@@')
#print(rand00)

for j in range(1):

    TD_model = Sequential()
    #TD_model.add(SimpleRNN(256,return_sequences=True, input_dim=INPUT_DIM, input_length=INPUT_LEN)) #256
    TD_model.add(LSTM(256,return_sequences=True, input_dim=INPUT_DIM, input_length=INPUT_LEN)) #256
    TD_model.add(Dropout(0.5))
    TD_model.add(LSTM(256,return_sequences=True ))
    TD_model.add(Dropout(0.5))
    TD_model.add(TimeDistributed(Dense(OUTPUT_LEN)))
#    TD_model.add(TimeDistributed(Dense(2)))
#    TD_model.add(Dense(4))
    TD_model.add(Activation('softmax'))


     BATCH_SIZE = 32
        
    ind_test_0 = []
    ind_train_0 = []
    ind_val_0 = []

    ind_test_0 = list(set(ind_tot))
    X_test1 = []


    X_test1 = X_0[ind_test_0,:,:]


    sgd = SGD(lr=0.01, decay=0.005, momentum=0.9, nesterov=True)

    TD_model.load_weights("./tmp/weights_SNP_7_eachframe_tot_tot.hdf5")
    TD_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    classes = TD_model.predict_classes(X_test1)
    print(classes)

    clas=np.ones(455)*6
    clas1=np.ones(22)*6
    for nf in range(0, 455):
        count=0
        for kf in range(22) :
            clas1[kf]=classes[nf,kf]
        clas1=np.array(clas1, dtype=np.int64)
        counts = np.bincount(clas1)
#        print(counts)
        clas[nf]=np.argmax(counts)
#        print(clas[nf])
#        if clas[nf]==nytt[nf]:
#            clas[nf]=nytt[nf]

    print('$$$$$$$$$$$$$$$$$$$$$')
    print(clas)
    cl= TD_model.predict(X_test1)
    print(cl)
    cl_p= TD_model.predict_proba(X_test1)
    print(cl_p[450])
    print(cl_p[450,1])



with open('prob_test_acc_tot1.csv','a') as f:
#    print('salaam')
  
    let1= array.array('c', ['\0' for _ in xrange(455)]) 
    for lk in range(0,455):
        if letter[lk]==1:
            let1[lk]='A'
        elif letter[lk]==2:
            let1[lk]='B'
        elif letter[lk]==3:
            let1[lk]='C'


    writer = csv.writer(f)
##    writer.writerow(('BVZ','Tray_label_single','Plant_label_single','Plant_label_single_letter', 'accession','1: Sf2', '2: Cvi', '3: Lan', '4: Col'))
    for nff in range (0,455):
        with open('id_72.csv') as id_72_1, open('id_acc_72.csv') as id_acc_72_1, open('position_72.csv') as pos_72_1, open('id_73.csv') as id_73_1, open('id_acc_73.csv') as id_acc_73_1, open('position_73.csv') as pos_73_1:
            id_72 = csv.DictReader(id_72_1)
            id_acc_72 = csv.DictReader(id_acc_72_1)
            pos_72 = csv.DictReader(pos_72_1)
            id_73 = csv.DictReader(id_73_1)
            id_acc_73 = csv.DictReader(id_acc_73_1)
            pos_73 = csv.DictReader(pos_73_1)
            nn=1000
            print(BVZ[nff])
            if BVZ[nff]==72:
                aa=0
                for row in pos_72:
                    hh=row['poss_72']
                    if len(hh)==3:
                        if int(hh[0])==Tray[nff] and  hh[1]== let1[nff] and int(hh[2])==y1[nff]:
                            print('h11')
                            nn=aa
                    if len(hh)==4:
                        if int(hh[0:2])==Tray[nff] and hh[2]== let1[nff] and int(hh[3])==y1[nff]:
                            nn=aa
                    aa=aa+1
                bb=0
                for row1 in id_acc_72:
                    if bb==nn:
                        kht=row1['idd_acc_72']
                    bb=bb+1 
            if BVZ[nff]==73:
#                print('h4')
                aa=0
                for row in pos_73:
                    hh=row['poss_73']
                    if len(hh)==3:
                        if int(hh[0])==Tray[nff] and hh[1]== let1[nff] and int(hh[2])==y1[nff]:
                            print('h5')
                            nn=aa
                    if len(hh)==4:
                        if  hh[2]==let1[nff] and int(hh[0:2])==Tray[nff] and int(hh[3])==y1[nff]:
                            nn=aa
                    aa=aa+1
                bb=0
                for row1 in id_acc_73:
                    if bb==nn:
                        kht=row1['idd_acc_73']
                    bb=bb+1
           
            print(nff)
##            writer.writerow((BVZ[nff],Tray[nff],y1[nff],let1[nff], kht))
            for nt in range (0,22):
                writer.writerow((cl_p[nff,nt]))

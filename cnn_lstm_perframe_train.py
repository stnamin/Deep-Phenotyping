##### This code is developed to extract deep features using cnn, train lstm with all the data  ####

from __future__ import print_function
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
import random
import glob
import csv
from sklearn.metrics import confusion_matrix
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

data=np.load('BVZ0072_BVZ0073_97pots_with_area.npz')
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
CNN_model = convnet('alexnet24',weights_path="./tmp/weights_alex_7_tot.hdf5", heatmap=False)
#        CNN_model = convnet('vgg_16',weights_path="vgg16_weights.h5", heatmap=False)


from keras import backend as K
get_dense_features = K.function([CNN_model.layers[0].input, K.learning_phase()],[CNN_model.layers[31].output]) #33

print ("Feature extraction...")
X_0 = np.zeros((97, INPUT_LEN, INPUT_DIM))
X_90 = np.zeros((97, INPUT_LEN, INPUT_DIM))
X_180 = np.zeros((97, INPUT_LEN, INPUT_DIM))
X_270 = np.zeros((97, INPUT_LEN, INPUT_DIM))
y = np.zeros((97, INPUT_LEN, OUTPUT_LEN))

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

        y[newcounter, inp_len, GL[newcounter*22+10]-1] = 1
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
ind_tot = range(97)
random.seed(16)
rand00=random.sample(ind_tot,97)
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


    ind_val_0 = rand00[0:15]

    ind_train_0 = list(set(ind_tot)- set(ind_val_0) )


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

    ny= np.zeros((newYval.shape[0], INPUT_LEN))
    for nf in range(0, newYval.shape[0]):
        for kf in range(22) :
            for df in range(4) :
                if newYval[nf,kf,df]==1:
                    ny[nf,kf]=df
    print(ny)
    checkpointer = ModelCheckpoint(filepath="./tmp/weights_SNP_7_eachframe_tot_tot.hdf5", verbose=1, save_best_only=True)
 
    sgd = SGD(lr=0.01, decay=0.005, momentum=0.9, nesterov=True)
    TD_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    TD_model.fit(X_train1, newYtrain, batch_size=32, nb_epoch=50, validation_data = (X_val1, newYval), callbacks=[checkpointer], sample_weight= None)

#            LSTM_model.save_weights("./tmp/final_weights_SNP1.hdf5",overwrite=True)
    TD_model.save_weights("./tmp/final_weights_SNP_7_eachframe_tot_tot.hdf5",overwrite=True)

    loss_and_metrics = TD_model.evaluate(X_val1, newYval)

    classes= np.array([])
    classes = TD_model.predict_classes(X_val1)
    score = TD_model.evaluate(X_val1, newYval, batch_size=BATCH_SIZE)
    print(score)
    result1= score[1]


    classes = TD_model.predict_classes(X_test1)
    score = TD_model.evaluate(X_test1, newYtest, batch_size=BATCH_SIZE)
    print(score)
    result2= score[1]
#    print(newYtest)
    nytt=np.zeros(newYtest.shape[0])
    for nf in range(0, newYtest.shape[0]):
        for df in range(4):
            if newYtest[nf,0,df]==1:
                nytt[nf]=df
    ny= np.zeros((newYtest.shape[0], INPUT_LEN))
#    ny=np.zeros(newYtest.shape[0])
    for nf in range(0, newYtest.shape[0]):
        for kf in range(22) :
            for df in range(4) :
                if newYtest[nf,kf,df]==1:
                    ny[nf,kf]=df


    TD_model.load_weights("./tmp/weights_SNP_7_eachframe_tot_tot.hdf5")
    TD_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    loss_and_metrics = TD_model.evaluate(X_test1, newYtest)
    classes = TD_model.predict_classes(X_test1)


    clas=np.ones(newYtest.shape[0])*6
    clas1=np.ones(22)*6
    for nf in range(0, newYtest.shape[0]):
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
    print('$$$$$$$$$$$$$$$$$$$$$')
    print(nytt)


    score = TD_model.evaluate(X_test1, newYtest, batch_size=BATCH_SIZE)
    print(score)
    f_matrix3 = confusion_matrix(nytt, clas)
    print(np.where(clas != nytt)[0])
    print((np.where(clas != nytt)[0])%20)
    print((np.where(clas != nytt)[0])/20)
    ind_test_0=np.array(ind_test_0)
    print(ind_test_0[(np.where(classes != ny)[0])%20])
    print(Tray[ind_test_0[(np.where(classes != ny)[0])%20]])
    print(BVZ[ind_test_0[(np.where(classes != ny)[0])%20]])
    print(y1[ind_test_0[(np.where(classes != ny)[0])%20]]-1)


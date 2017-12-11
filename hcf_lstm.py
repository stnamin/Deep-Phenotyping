##### This code is developed to use extracted handcrafted features, train and test with lstm ####

from __future__ import print_function
from keras.models import Sequential,Model
from keras.layers import LSTM
import matplotlib.pyplot as plt
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
from skimage.feature import greycomatrix, greycoprops
import glob
import csv
from sklearn.metrics import confusion_matrix
np.set_printoptions(threshold=np.nan)
from sklearn import preprocessing
from skimage.color import rgb2gray
#########################################################
#### Reading the images
#########################################################
INPUT_LEN = 22
INPUT_DIM = 1050
OUTPUT_LEN = 4

batch_size = 32
nb_classes = 4
data_augmentation = False

# input image dimensions
img_rows, img_cols = 320, 320
# the images are RGB
img_channels = 3

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

contrast=np.zeros((TOT_0.shape[0],4),dtype=np.uint8)
energy=np.zeros((TOT_0.shape[0],4),dtype=np.uint8)
homogeneity=np.zeros((TOT_0.shape[0],4),dtype=np.uint8)
for ii in range (TOT_0.shape[0]):
    LEVELS = 17
    img00 = LEVELS * rgb2gray(TOT_0[ii,:,:,:])
    rr0 = greycomatrix(img00, [1], [0,np.pi/4, np.pi/2, 3*np.pi/4], levels = LEVELS) #np.pi/4, np.pi/2, 3*np.pi/4], levels=16)
    rr01=rr0[1:,1:,:,:]
    contrast[ii,0:4] = greycoprops(rr01, 'contrast')
    energy[ii,0:4] = greycoprops(rr01, 'energy')
    homogeneity[ii,0:4] = greycoprops(rr01, 'homogeneity')


TOT_0=np.transpose(TOT_0,(0,3,1,2))
TOT_90=np.transpose(TOT_90,(0,3,1,2))
TOT_180=np.transpose(TOT_180,(0,3,1,2))
TOT_270=np.transpose(TOT_270,(0,3,1,2))

#TOT_0 = TOT_0.astype('float32')
#TOT_90 = TOT_90.astype('float32')
#TOT_180 = TOT_180.astype('float32')
#TOT_270 = TOT_270.astype('float32')
#X_train /= 255
#X_test /= 255
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


np.savez('Data_7_97f_ba_glcm', TOT_0_r=TOT_0_r, TOT_90_r=TOT_90_r, TOT_180_r=TOT_180_r, TOT_270_r=TOT_270_r, GL=GL, y1=y1, Tray=Tray, BVZ=BVZ, contrast=contrast, energy=energy, homogeneity=homogeneity)

fea=np.load('BVZ0072_BVZ0073_97pots_features.npz')
area=fea['Area']
compactness=fea['Compactness']
eccentricity=fea['Eccentricity']
extent=fea['Extent']
max_b=fea['Max_B']
max_g=fea['Max_G']
max_r=fea['Max_R']
min_b=fea['Min_B']
min_g=fea['Min_G']
min_r=fea['Min_R']
mean_b=fea['Mean_B']
mean_g=fea['Mean_G']
mean_r=fea['Mean_R']
mean_h=fea['Mean_H']
mean_s=fea['Mean_S']
mean_v=fea['Mean_V']
perimeter=fea['Perimeter']
roundness=fea['Roundness']
fea1=np.load('BVZ0072_BVZ0073_97pots_fourier_features.npz')
fou_X=fea1['FOURIER_X_real']
fou_XI=fea1['FOURIER_X_imag']
fou_YI=fea1['FOURIER_Y_imag']
fou_Y=fea1['FOURIER_Y_real']
print(fou_X.dtype)

fou_RX=abs(fou_X+fou_XI*1j)
print(fou_RX.shape)
print(fou_X[1,1])
print(fou_XI[1,1])
print(fou_RX[1,1])
fou_RY=abs(fou_Y+fou_YI*1j)

min_max_scaler = preprocessing.MinMaxScaler()
area = min_max_scaler.fit_transform(area)
compactness = min_max_scaler.fit_transform(compactness)
eccentricity = min_max_scaler.fit_transform(eccentricity)
extent = min_max_scaler.fit_transform(extent)
max_b = min_max_scaler.fit_transform(max_b)
max_g = min_max_scaler.fit_transform(max_g)
max_r = min_max_scaler.fit_transform(max_r)
min_b = min_max_scaler.fit_transform(min_b)
min_g = min_max_scaler.fit_transform(min_g)
min_r = min_max_scaler.fit_transform(min_r)
mean_b = min_max_scaler.fit_transform(mean_b)
mean_g = min_max_scaler.fit_transform(mean_g)
mean_r = min_max_scaler.fit_transform(mean_r)
mean_h = min_max_scaler.fit_transform(mean_h)
mean_s = min_max_scaler.fit_transform(mean_s)
mean_v = min_max_scaler.fit_transform(mean_v)
perimeter = min_max_scaler.fit_transform(perimeter)
roundness = min_max_scaler.fit_transform(roundness)

print(fou_RX[0,:])

fou_RX = min_max_scaler.fit_transform(fou_RX)
fou_RY = min_max_scaler.fit_transform(fou_RY)
contrast = min_max_scaler.fit_transform(contrast)
energy = min_max_scaler.fit_transform(energy)
homogeneity = min_max_scaler.fit_transform(homogeneity)

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
     

X_0 = np.zeros((97, INPUT_LEN, INPUT_DIM))
X_90 = np.zeros((97, INPUT_LEN, INPUT_DIM))
X_180 = np.zeros((97, INPUT_LEN, INPUT_DIM))
X_270 = np.zeros((97, INPUT_LEN, INPUT_DIM))
y = np.zeros((97, INPUT_LEN, OUTPUT_LEN))

for newcounter in range(0, 97):
    for inp_len in range (0, 22):
       
#        current_frame = np.zeros((1,3,227,227))
#        current_frame = np.array(np.expand_dims(TOT_0_r[newcounter*22+inp_len,:,:,:], axis=0))
        X_0[newcounter, inp_len,0] = area[newcounter*22+inp_len]
        X_0[newcounter, inp_len,1] = compactness[newcounter*22+inp_len]
        X_0[newcounter, inp_len,2] = eccentricity[newcounter*22+inp_len]
        X_0[newcounter, inp_len,3] = extent[newcounter*22+inp_len]
        X_0[newcounter, inp_len,4] = max_b[newcounter*22+inp_len]
        X_0[newcounter, inp_len,5] = max_g[newcounter*22+inp_len]
        X_0[newcounter, inp_len,6] = max_r[newcounter*22+inp_len]
        X_0[newcounter, inp_len,7] = min_b[newcounter*22+inp_len]
        X_0[newcounter, inp_len,8] = min_g[newcounter*22+inp_len]
        X_0[newcounter, inp_len,9] = min_r[newcounter*22+inp_len]
        X_0[newcounter, inp_len,10] = mean_b[newcounter*22+inp_len]
        X_0[newcounter, inp_len,11] = mean_g[newcounter*22+inp_len]
        X_0[newcounter, inp_len,12] = mean_r[newcounter*22+inp_len]
        X_0[newcounter, inp_len,13] = mean_h[newcounter*22+inp_len]
        X_0[newcounter, inp_len,14] = mean_s[newcounter*22+inp_len]
        X_0[newcounter, inp_len,15] = mean_v[newcounter*22+inp_len]
        X_0[newcounter, inp_len,16] = perimeter[newcounter*22+inp_len]
        X_0[newcounter, inp_len,17] = roundness[newcounter*22+inp_len]
        X_0[newcounter, inp_len,18:273] = fou_X[newcounter*22+inp_len,:]
        X_0[newcounter, inp_len,273:528] = fou_XI[newcounter*22+inp_len,:]
        X_0[newcounter, inp_len,528:783] = fou_YI[newcounter*22+inp_len,:]
        X_0[newcounter, inp_len,783:1038] = fou_Y[newcounter*22+inp_len,:]
        X_0[newcounter, inp_len,1038:1042] = contrast[newcounter*22+inp_len,0:4]
        X_0[newcounter, inp_len,1042:1046] = energy[newcounter*22+inp_len,0:4]
        X_0[newcounter, inp_len,1046:1050] = homogeneity[newcounter*22+inp_len,0:4]

        y[newcounter, inp_len, GL[newcounter*22+10]-1] = 1
    print (newcounter)
print(X_0.dtype)       
#########################################################
#### Preparing the data for training the LSTM model
#########################################################
######################################################
#### LSTM Model
#########################################################
INPUT_LEN = 22
INPUT_DIM =1050  #31#18
OUTPUT_LEN = 4
ind_tot = range(97)
random.seed(16)
rand00=random.sample(ind_tot,97)


for j in range(5):

    TD_model = Sequential()
    TD_model.add(LSTM(256,return_sequences=True, input_dim=INPUT_DIM, input_length=INPUT_LEN)) #256
    TD_model.add(Dropout(0.5))
    TD_model.add(LSTM(256,return_sequences=True ))
    TD_model.add(Dropout(0.5))
#LSTM_model.add(TimeDistributed(Dense(2)))
    TD_model.add(TimeDistributed(Dense(OUTPUT_LEN)))
#    TD_model.add(Dense(4))
    TD_model.add(Activation('softmax'))

#            sgd = SGD(lr=0.01, decay=0.005 , momentum=0.9, nesterov=True)
#            TD_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
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


    X_test1 = np.concatenate((X_0[ind_test_0,:,:],X_0[ind_test_90,:,:],X_0[ind_test_180,:,:],X_0[ind_test_270,:,:]),axis=0)
    X_val1 = np.concatenate((X_0[ind_val_0,:,:],X_0[ind_val_90,:,:],X_0[ind_val_180,:,:],X_0[ind_val_270,:,:]),axis=0)
    X_train1 = np.concatenate((X_0[ind_train_0,:,:],X_0[ind_train_90,:,:],X_0[ind_train_180,:,:],X_0[ind_train_270,:,:]),axis=0)
    newYtest=np.concatenate((y[ind_test_0],y[ind_test_90],y[ind_test_180],y[ind_test_270]),axis=0)
    newYval=np.concatenate((y[ind_val_0],y[ind_val_90],y[ind_val_180],y[ind_val_270]),axis=0)
    newYtrain=np.concatenate((y[ind_train_0],y[ind_train_90],y[ind_train_180],y[ind_train_270]),axis=0)

    ny= np.zeros((newYval.shape[0], INPUT_LEN))
#    ny=np.zeros(newYval.shape[0])
    for nf in range(0, newYval.shape[0]):
        for kf in range(22) :
            for df in range(4) :
                if newYval[nf,kf,df]==1:
                    ny[nf,kf]=df
    print(ny)
    checkpointer = ModelCheckpoint(filepath="./tmp/weights_SNP_7_eachframe.hdf5", verbose=1, save_best_only=True)

    sgd = SGD(lr=0.01, decay=0.005, momentum=0.9, nesterov=True)
    TD_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    TD_model.fit(X_train1, newYtrain, batch_size=32, nb_epoch=100, validation_data = (X_val1, newYval), callbacks=[checkpointer], sample_weight= None)

#            LSTM_model.save_weights("./tmp/final_weights_SNP1.hdf5",overwrite=True)
    TD_model.save_weights("./tmp/final_weights_SNP_7_eachframe.hdf5",overwrite=True)

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


    TD_model.load_weights("./tmp/weights_SNP_7_eachframe.hdf5")
    TD_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    loss_and_metrics = TD_model.evaluate(X_test1, newYtest)
    classes = TD_model.predict_classes(X_test1)
    print(classes)



    clas=np.ones(newYtest.shape[0])*6
    clas1=np.ones(22)*6
    for nf in range(0, newYtest.shape[0]):
        count=0
        for kf in range(22) :
            clas1[kf]=classes[nf,kf]
        clas1=np.array(clas1, dtype=np.int64)
        counts = np.bincount(clas1)
        print(counts)
        clas[nf]=np.argmax(counts)
        print(clas[nf])

    print('$$$$$$$$$$$$$$$$$$$$$')
    print(clas)
    print('$$$$$$$$$$$$$$$$$$$$$')
    print(nytt)


    f_matrix3 = confusion_matrix(nytt, clas)
    print(f_matrix3)
    print(np.where(clas != nytt)[0])
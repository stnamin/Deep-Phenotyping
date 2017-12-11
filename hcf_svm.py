##### This code is developed to use extracted handcrafted features and svm for classification ####

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
from sklearn.svm import LinearSVC
#from pystruct.models import ChainCRF
#from pystruct.learners import FrankWolfeSSVM

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
# the  images are RGB
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
#    print(np.max(img00))
    #plt.imshow(TOT_0[ii,:,:,:])
    #plt.imshow(img00)
    #plt.show()
    rr0 = greycomatrix(img00, [1], [0,np.pi/4, np.pi/2, 3*np.pi/4], levels = LEVELS) #np.pi/4, np.pi/2, 3*np.pi/4], levels=16)
#    print(rr0.shape)
#    print(rr0.T)
#    rr01=np.zeros[7,7,1,4]i
    rr01=rr0[1:,1:,:,:]
#    print(rr01.shape)
#    print(rr01.T)
    contrast[ii,0:4] = greycoprops(rr01, 'contrast')
#    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
#    print(contrast)
    energy[ii,0:4] = greycoprops(rr01, 'energy')
    homogeneity[ii,0:4] = greycoprops(rr01, 'homogeneity')
#    print(energy)
#    print(homogeneity)

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
        X_0[newcounter, inp_len, 18:273] = fou_X[newcounter * 22 + inp_len, :]
        X_0[newcounter, inp_len, 273:528] = fou_XI[newcounter * 22 + inp_len, :]
        X_0[newcounter, inp_len, 528:783] = fou_YI[newcounter * 22 + inp_len, :]
        X_0[newcounter, inp_len, 783:1038] = fou_Y[newcounter * 22 + inp_len, :]
        X_0[newcounter, inp_len, 1038:1042] = contrast[newcounter * 22 + inp_len, 0:4]
        X_0[newcounter, inp_len, 1042:1046] = energy[newcounter * 22 + inp_len, 0:4]
        X_0[newcounter, inp_len, 1046:1050] = homogeneity[newcounter * 22 + inp_len, 0:4]

        y[newcounter, inp_len, GL[newcounter*22+10]-1] = 1
    print (newcounter)
print(X_0.dtype)
XXX=np.zeros((97*22, INPUT_DIM))
kk=0
for ii in range (0,97):
    for jj in range (0,22):
        XXX[kk]=X_0[ii,jj,:]
        kk=kk+1
   
#########################################################
#### Preparing the data for training SVM
#########################################################
######################################################
#########################################################
INPUT_LEN = 22
INPUT_DIM =1050
OUTPUT_LEN = 4
ind_tot = range(97)
random.seed(16)
rand00=random.sample(ind_tot,97)


for j in range(5):

    BATCH_SIZE = 32          
        
    ind_test_00 = []
    ind_train_00 = []
    ind_val_00 = []

    if j!= 4 :
        ind_test_00 = rand00[j*20:(j+1)*20]
        ind_val_00 = rand00[82:97]
    else :
        ind_test_00 = rand00[77:97]
        ind_val_00 = rand00[0:15]

    ind_train_00 = list(set(ind_tot) - set(ind_test_00) - set(ind_val_00))

    ind_test_0=np.array([])
    ind_val_0=np.array([])
    ind_train_0=np.array([])

    for i in ind_test_00:
        ind_test_0=np.append(ind_test_0,np.arange(i*22,(i+1)*22))
    for i in ind_val_00:
        ind_val_0=np.append(ind_val_0,np.arange(i*22,(i+1)*22))
    for i in ind_train_00:
        ind_train_0=np.append(ind_train_0,np.arange(i*22,(i+1)*22))

    ind_test_0=np.uint16(ind_test_0)
    ind_val_0=np.uint16(ind_val_0)
    ind_train_0=np.uint16(ind_train_0)



    ind_test_90 = []  
    ind_train_90 = []
    ind_val_90 = []
    ind_test_180 = []  
    ind_train_180 = []   
    ind_val_180 = []
    ind_test_270 = []  
    ind_train_270 = []   
    ind_val_270 = []


    newYtrain = []
    newYtest = []
    newYval = []
    X_train1 = []
    X_val1 = []
    X_test1 = []

    X_test1 = XXX[ind_test_0,:]
    X_val1 = XXX[ind_val_0,:]
    X_train1 = XXX[ind_train_0,:]

    newYtest=GL[ind_test_0]
    newYval=GL[ind_val_0]
    newYtrain=GL[ind_train_0]


    # Train linear SVM
    svm = LinearSVC(dual=False, C=.1)
#    svm.fit(np.vstack(X_train1), np.hstack(newYtrain))
    svm.fit((X_train1),(newYtrain))
    out_s = svm.predict((X_test1))
    print(out_s)
    print(newYtest)
    pp1=0
    pp2=0
    pp3=0
    pp4=0
    for nf in range(0, newYtest.shape[0]):
        if newYtest[nf]==1:
            pp1=pp1+1
        if newYtest[nf]==2:
            pp2=pp2+1
        if newYtest[nf]==3:
            pp3=pp3+1
        if newYtest[nf]==4:
            pp4=pp4+1
    rr1=0
    rr2=0
    rr3=0
    rr4=0
    for nf in range(0, newYtest.shape[0]):
        if newYtest[nf]==1 and out_s[nf]==1:
            rr1=rr1+1
        if newYtest[nf]==2 and out_s[nf]==2:
            rr2=rr2+1
        if newYtest[nf]==3 and out_s[nf]==3:
            rr3=rr3+1
        if newYtest[nf]==4 and out_s[nf]==4:
            rr4=rr4+1
    print(pp1)
    print(pp2)
    print(pp3)
    print(pp4)

    print(rr1)
    print(rr2)
    print(rr3)
    print(rr4)
    print(svm.score((X_test1), (newYtest)))
    print(newYtest)
    print(out_s)
    f_matrix1 = confusion_matrix(np.hstack(newYtest), out_s)
    print(f_matrix1)


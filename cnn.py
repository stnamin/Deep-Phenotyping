##### This code is developed for train and test with cnn (alexnet)  ####

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

TOT_0=data['TOT_0']
TOT_90=data['TOT_90']
TOT_180=data['TOT_180']
TOT_270=data['TOT_270']
GL=data['Plant_label']
y1=data['Plant_label_single']
area=data['Area']

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


np.savez('Data_7', TOT_0_r=TOT_0_r, TOT_90_r=TOT_90_r, TOT_180_r=TOT_180_r, TOT_270_r=TOT_270_r, GL=GL, y1=y1, area=area)

print(area)
print(y1)

lab1=np.sum(y1==1)
lab2=np.sum(y1==2)
lab3=np.sum(y1==3)
lab4=np.sum(y1==4)
print(np.sum(y1==1))
print(np.sum(y1==2))
print(np.sum(y1==3))
print(np.sum(y1==4))
print(GL.shape[0])
print(TOT_0_r.shape[0])
     
#########################################################
INPUT_LEN = 22
INPUT_DIM = 4096
OUTPUT_LEN = 24
ind_tot = range(97)

BATCH_SIZE = 32          
        
ind_test_00 = []
ind_train_00 = []
ind_val_00 = []

random.seed(16)
rand00=random.sample(ind_tot,97)
ind_test_00 = rand00[k*20:(k+1)*20] #k=0,1,2,3,4 for 5 folds
ind_val_00 = rand00[82:97]

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
     
ind_test_90 = ind_test_0
ind_val_90 = ind_val_0 
ind_train_90 = ind_train_0
ind_test_180 = ind_test_0
ind_val_180 = ind_val_0 
ind_train_180 = ind_train_0 
ind_test_270 = ind_test_0 
ind_val_270 = ind_val_0
ind_train_270 = ind_train_0
          
#print(ind_test_0)
#print(ind_test_90)
#print(ind_test_180)
#print(ind_test_270)          

newYtrain = []
newYtest = []
newYval = []
X_train1 = []
X_val1 = []
X_test1 = []

y2 = np.zeros(GL.shape[0])
for nh in range(GL.shape[0]):
    if area[nh] <= 7000 and GL[nh]==1:
        y2[nh]=0
    if area[nh] > 7000 and area[nh] <= 14000 and GL[nh]==1:
        y2[nh]=1
    if area[nh] > 14000 and area[nh] <= 21000 and GL[nh]==1:
        y2[nh]=2
    if area[nh] > 21000 and area[nh] <= 28000 and GL[nh]==1:
        y2[nh]=3
    if area[nh] > 28000 and area[nh] <= 35000 and GL[nh]==1:
        y2[nh]=4
    if area[nh] > 35000 and GL[nh]==1:
        y2[nh]=5
    if area[nh] <= 7000 and GL[nh]==2:
        y2[nh]=6
    if area[nh] > 7000 and area[nh] <= 14000 and GL[nh]==2:
        y2[nh]=7
    if area[nh] > 14000 and area[nh] <= 21000 and GL[nh]==2:
        y2[nh]=8
    if area[nh] > 21000 and area[nh] <= 28000 and GL[nh]==2:
        y2[nh]=9
    if area[nh] > 28000 and area[nh] <= 35000 and GL[nh]==2:
        y2[nh]=10
    if area[nh] > 35000 and GL[nh]==2:
        y2[nh]=11
    if area[nh] <= 7000 and GL[nh]==3:
        y2[nh]=12
    if area[nh] > 7000 and area[nh] <= 14000 and GL[nh]==3:
        y2[nh]=13
    if area[nh] > 14000 and area[nh] <= 21000 and GL[nh]==3:
        y2[nh]=14
    if area[nh] > 21000 and area[nh] <= 28000 and GL[nh]==3:
        y2[nh]=15
    if area[nh] > 28000 and area[nh] <= 35000 and GL[nh]==3:
        y2[nh]=16
    if area[nh] > 35000 and GL[nh]==3:
        y2[nh]=17
    if area[nh] <= 7000 and GL[nh]==4:
        y2[nh]=18
    if area[nh] > 7000 and area[nh] <= 14000 and GL[nh]==4:
        y2[nh]=19
    if area[nh] > 14000 and area[nh] <= 21000 and GL[nh]==4:
        y2[nh]=20
    if area[nh] > 21000 and area[nh] <= 28000 and GL[nh]==4:
        y2[nh]=21
    if area[nh] > 28000 and area[nh] <= 35000 and GL[nh]==4:
        y2[nh]=22
    if area[nh] > 35000 and GL[nh]==4:
        y2[nh]=23


y = np.zeros((GL.shape[0], OUTPUT_LEN))

for mh in range(GL.shape[0]):
    y[mh,int(y2[mh])]=1


X_test1 = np.concatenate((TOT_0_r[ind_test_0,:,:],TOT_90_r[ind_test_90,:,:],TOT_180_r[ind_test_180,:,:],TOT_270_r[ind_test_270,:,:]),axis=0)
X_val1 = np.concatenate((TOT_0_r[ind_val_0,:,:],TOT_90_r[ind_val_90,:,:],TOT_180_r[ind_val_180,:,:],TOT_270_r[ind_val_270,:,:]),axis=0)
X_train1 = np.concatenate((TOT_0_r[ind_train_0,:,:],TOT_90_r[ind_train_90,:,:],TOT_180_r[ind_train_180,:,:],TOT_270_r[ind_train_270,:,:]),axis=0)
newYtest=np.concatenate((y[ind_test_0],y[ind_test_90],y[ind_test_180],y[ind_test_270]),axis=0)
newYval=np.concatenate((y[ind_val_0],y[ind_val_90],y[ind_val_180],y[ind_val_270]),axis=0)
newYtrain=np.concatenate((y[ind_train_0],y[ind_train_90],y[ind_train_180],y[ind_train_270]),axis=0)

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
###model = convnet('alexnet',weights_path="alexnet_weights.h5", heatmap=False)
###model2 = convnet('alexnet24', weights_path=None, heatmap=False)
model2 = convnet('alexnet24', weights_path="./tmp/weights_alex_7_5.hdf5", heatmap=False)###

model2.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

ny=np.zeros(newYtest.shape[0])
for nf in range(0, newYtest.shape[0]):
    for df in range(24):
        if newYtest[nf,df]==1:
            ny[nf]=df


score = model2.evaluate(X_test1, newYtest, batch_size=32)
loss_and_metrics = model2.evaluate(X_test1, newYtest)
predict = model2.predict(X_test1, batch_size=32)
print(predict)
ind=np.argmax(predict,1)

f_matrix = confusion_matrix(ny, ind)
print(f_matrix)

ind2=np.argmax(newYtest,1)
print(ind2[0:120])
print(ind[0:120])
print(ny[0:120])
pp1=0
pp2=0
pp3=0
pp4=0
for nf in range(0, newYtest.shape[0]):
    if ind2[nf]==0 or ind2[nf]==1 or ind2[nf]==2 or ind2[nf]==3 or ind2[nf]==4 or ind2[nf]==5 :
        pp1=pp1+1 
    if ind2[nf]==6 or ind2[nf]==7 or ind2[nf]==8 or ind2[nf]==9 or ind2[nf]==10 or ind2[nf]==11 :
        pp2=pp2+1
    if ind2[nf]==12 or ind2[nf]==13 or ind2[nf]==14 or ind2[nf]==15 or ind2[nf]==16 or ind2[nf]==17 :
        pp3=pp3+1
    if ind2[nf]==18 or ind2[nf]==19 or ind2[nf]==20 or ind2[nf]==21 or ind2[nf]==22 or ind2[nf]==23 :
        pp4=pp4+1


rr1=0
rr2=0
rr3=0
rr4=0
for nf in range(0, newYtest.shape[0]):
    if (ind[nf]==0 or ind[nf]==1 or ind[nf]==2 or ind[nf]==3 or ind[nf]==4 or ind[nf]==5) and (ny[nf]==0 or ny[nf]==1 or ny[nf]==2 or ny[nf]==3 or ny[nf]==4 or ny[nf]==5):
        rr1=rr1+1
    if (ind[nf]==6 or ind[nf]==7 or ind[nf]==8 or ind[nf]==9 or ind[nf]==10 or ind[nf]==11) and (ny[nf]==6 or ny[nf]==7 or ny[nf]==8 or ny[nf]==9 or ny[nf]==10 or ny[nf]==11):
        rr2=rr2+1
    if (ind[nf]==12 or ind[nf]==13 or ind[nf]==14 or ind[nf]==15 or ind[nf]==16 or ind[nf]==17) and (ny[nf]==12 or ny[nf]==13 or ny[nf]==14 or ny[nf]==15 or ny[nf]==16 or ny[nf]==17):
        rr3=rr3+1
    if (ind[nf]==18 or ind[nf]==19 or ind[nf]==20 or ind[nf]==21 or ind[nf]==22 or ind[nf]==23) and (ny[nf]==18 or ny[nf]==19 or ny[nf]==20 or ny[nf]==21 or ny[nf]==22 or ny[nf]==23):
        rr4=rr4+1


print(pp1)
print(pp2)
print(pp3)
print(pp4)

print(rr1)
print(rr2)
print(rr3)
print(rr4)
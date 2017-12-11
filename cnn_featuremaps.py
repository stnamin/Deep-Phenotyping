##### This code is developed to demonstrate feature maps form differnt layers of cnn  ####

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
import glob
import csv
from sklearn.metrics import confusion_matrix
np.set_printoptions(threshold=np.nan)
from sklearn.svm import LinearSVC
from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM
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

data2=np.load('BVZ0072_BVZ0073_97pots.npz')
print('hey')

TOT_000=data2['TOT_0']

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
CNN_model = convnet('alexnet24',weights_path="./tmp/weights_alex_7_5.hdf5", heatmap=False)
#        CNN_model = convnet('vgg_16',weights_path="vgg16_weights.h5", heatmap=False)


from keras import backend as K
get_dense_features26 = K.function([CNN_model.layers[0].input, K.learning_phase()],[CNN_model.layers[26].output]) #31
get_dense_features25 = K.function([CNN_model.layers[0].input, K.learning_phase()],[CNN_model.layers[25].output])
get_dense_features24 = K.function([CNN_model.layers[0].input, K.learning_phase()],[CNN_model.layers[24].output])
get_dense_features23 = K.function([CNN_model.layers[0].input, K.learning_phase()],[CNN_model.layers[23].output])
get_dense_features22 = K.function([CNN_model.layers[0].input, K.learning_phase()],[CNN_model.layers[22].output])
get_dense_features21 = K.function([CNN_model.layers[0].input, K.learning_phase()],[CNN_model.layers[21].output])
get_dense_features20 = K.function([CNN_model.layers[0].input, K.learning_phase()],[CNN_model.layers[20].output])
get_dense_features19 = K.function([CNN_model.layers[0].input, K.learning_phase()],[CNN_model.layers[19].output])
get_dense_features18 = K.function([CNN_model.layers[0].input, K.learning_phase()],[CNN_model.layers[18].output])
get_dense_features17 = K.function([CNN_model.layers[0].input, K.learning_phase()],[CNN_model.layers[17].output])
get_dense_features16 = K.function([CNN_model.layers[0].input, K.learning_phase()],[CNN_model.layers[16].output])
get_dense_features15 = K.function([CNN_model.layers[0].input, K.learning_phase()],[CNN_model.layers[15].output])
get_dense_features14 = K.function([CNN_model.layers[0].input, K.learning_phase()],[CNN_model.layers[14].output])
get_dense_features13 = K.function([CNN_model.layers[0].input, K.learning_phase()],[CNN_model.layers[13].output])
get_dense_features12 = K.function([CNN_model.layers[0].input, K.learning_phase()],[CNN_model.layers[12].output])
get_dense_features11 = K.function([CNN_model.layers[0].input, K.learning_phase()],[CNN_model.layers[11].output])
get_dense_features10 = K.function([CNN_model.layers[0].input, K.learning_phase()],[CNN_model.layers[10].output])
get_dense_features9 = K.function([CNN_model.layers[0].input, K.learning_phase()],[CNN_model.layers[9].output])
get_dense_features8 = K.function([CNN_model.layers[0].input, K.learning_phase()],[CNN_model.layers[8].output])
get_dense_features7 = K.function([CNN_model.layers[0].input, K.learning_phase()],[CNN_model.layers[7].output])
get_dense_features6 = K.function([CNN_model.layers[0].input, K.learning_phase()],[CNN_model.layers[6].output])
get_dense_features5 = K.function([CNN_model.layers[0].input, K.learning_phase()],[CNN_model.layers[5].output])
get_dense_features4 = K.function([CNN_model.layers[0].input, K.learning_phase()],[CNN_model.layers[4].output])
get_dense_features3 = K.function([CNN_model.layers[0].input, K.learning_phase()],[CNN_model.layers[3].output])
get_dense_features2 = K.function([CNN_model.layers[0].input, K.learning_phase()],[CNN_model.layers[2].output])
get_dense_features1 = K.function([CNN_model.layers[0].input, K.learning_phase()],[CNN_model.layers[1].output])



print ("Feature extraction...")
X_0_26 = np.zeros((97, INPUT_LEN, 256, 6,6))
X_0_25 = np.zeros((97, INPUT_LEN, 256, 13,13))
X_0_24 = np.zeros((97, INPUT_LEN, 128, 13,13))
X_0_23 = np.zeros((97, INPUT_LEN, 128, 13,13))
X_0_22 = np.zeros((97, INPUT_LEN, 192, 15,15))
X_0_21 = np.zeros((97, INPUT_LEN, 192, 15,15))
X_0_20 = np.zeros((97, INPUT_LEN, 384, 15,15))
X_0_19 = np.zeros((97, INPUT_LEN, 384, 13,13))
X_0_18 = np.zeros((97, INPUT_LEN, 192, 13,13))
X_0_17 = np.zeros((97, INPUT_LEN, 192, 13,13))
X_0_16 = np.zeros((97, INPUT_LEN, 192, 15,15))
X_0_15 = np.zeros((97, INPUT_LEN, 192, 15,15))
X_0_14 = np.zeros((97, INPUT_LEN, 384, 15,15))
X_0_13 = np.zeros((97, INPUT_LEN, 384, 13,13))
X_0_12 = np.zeros((97, INPUT_LEN, 256, 15,15))
X_0_11 = np.zeros((97, INPUT_LEN, 256, 13,13))
X_0_10 = np.zeros((97, INPUT_LEN, 256, 13,13))
X_0_9 = np.zeros((97, INPUT_LEN, 256, 27,27))
X_0_8 = np.zeros((97, INPUT_LEN, 128, 27,27))
X_0_7 = np.zeros((97, INPUT_LEN, 128, 27,27))
X_0_6 = np.zeros((97, INPUT_LEN, 48, 31,31))
X_0_5 = np.zeros((97, INPUT_LEN, 48, 31, 31))
X_0_4 = np.zeros((97, INPUT_LEN, 96, 31,31))
X_0_3 = np.zeros((97, INPUT_LEN, 96, 27,27))
X_0_2 = np.zeros((97, INPUT_LEN, 96, 27,27))
X_0_1 = np.zeros((97, INPUT_LEN, 96, 55,55))


X_0 = [X_0_1, X_0_2, X_0_3, X_0_4, X_0_5, X_0_6, X_0_7, X_0_8, X_0_9, X_0_10, X_0_11, X_0_12, X_0_13, X_0_14, X_0_15, X_0_16, X_0_17, X_0_18, X_0_19, X_0_20, X_0_21, X_0_22, X_0_23, X_0_24, X_0_25, X_0_26]


#X_90 = np.zeros((97, INPUT_LEN, INPUT_DIM))
#X_180 = np.zeros((97, INPUT_LEN, INPUT_DIM))
#X_270 = np.zeros((97, INPUT_LEN, INPUT_DIM))
y = np.zeros((97, OUTPUT_LEN))

for newcounter in range(0, 97):
    for inp_len in range (0, 22):
       # continue
        current_frame = np.zeros((1,3,227,227))
        current_frame = np.array(np.expand_dims(TOT_0_r[newcounter*22+inp_len,:,:,:], axis=0))
        X_0_26[newcounter, inp_len] = np.array(get_dense_features26([current_frame,0]))[0]
        X_0_25[newcounter, inp_len] = np.array(get_dense_features25([current_frame,0]))[0]
        X_0_24[newcounter, inp_len] = np.array(get_dense_features24([current_frame,0]))[0]
        X_0_23[newcounter, inp_len] = np.array(get_dense_features23([current_frame,0]))[0]
        X_0_22[newcounter, inp_len] = np.array(get_dense_features22([current_frame,0]))[0]
        X_0_21[newcounter, inp_len] = np.array(get_dense_features21([current_frame,0]))[0]
        X_0_20[newcounter, inp_len] = np.array(get_dense_features20([current_frame,0]))[0]
        X_0_19[newcounter, inp_len] = np.array(get_dense_features19([current_frame,0]))[0]
        X_0_18[newcounter, inp_len] = np.array(get_dense_features18([current_frame,0]))[0]
        X_0_17[newcounter, inp_len] = np.array(get_dense_features17([current_frame,0]))[0]
        X_0_16[newcounter, inp_len] = np.array(get_dense_features16([current_frame,0]))[0]
        X_0_15[newcounter, inp_len] = np.array(get_dense_features15([current_frame,0]))[0]
        X_0_14[newcounter, inp_len] = np.array(get_dense_features14([current_frame,0]))[0]
        X_0_13[newcounter, inp_len] = np.array(get_dense_features13([current_frame,0]))[0]
        X_0_12[newcounter, inp_len] = np.array(get_dense_features12([current_frame,0]))[0]
        X_0_11[newcounter, inp_len] = np.array(get_dense_features11([current_frame,0]))[0]
        X_0_10[newcounter, inp_len] = np.array(get_dense_features10([current_frame,0]))[0]
        X_0_9[newcounter, inp_len] = np.array(get_dense_features9([current_frame,0]))[0]
        X_0_8[newcounter, inp_len] = np.array(get_dense_features8([current_frame,0]))[0]
        X_0_7[newcounter, inp_len] = np.array(get_dense_features7([current_frame,0]))[0]
        X_0_6[newcounter, inp_len] = np.array(get_dense_features6([current_frame,0]))[0]
        X_0_5[newcounter, inp_len] = np.array(get_dense_features5([current_frame,0]))[0]
        X_0_4[newcounter, inp_len] = np.array(get_dense_features4([current_frame,0]))[0]
        X_0_3[newcounter, inp_len] = np.array(get_dense_features3([current_frame,0]))[0]
        X_0_2[newcounter, inp_len] = np.array(get_dense_features2([current_frame,0]))[0]
        X_0_1[newcounter, inp_len] = np.array(get_dense_features1([current_frame,0]))[0]


    y[newcounter,GL[newcounter*22+10]-1] = 1
    print (newcounter)



np.savez('X_0_features', X_0_1=X_0_1, X_0_2=X_0_2, X_0_3=X_0_3, X_0_4=X_0_4, X_0_5=X_0_5, X_0_6=X_0_6, X_0_7=X_0_7, X_0_8=X_0_8, X_0_9=X_0_9, X_0_10=X_0_10, X_0_11=X_0_11, X_0_12=X_0_12, X_0_13=X_0_13, X_0_14=X_0_14, X_0_15=X_0_15, X_0_16=X_0_16, X_0_17=X_0_17, X_0_18=X_0_18, X_0_19=X_0_19, X_0_20=X_0_20, X_0_21=X_0_21, X_0_22=X_0_22, X_0_23=X_0_23, X_0_24=X_0_24, X_0_25=X_0_25, X_0_26=X_0_26)
exit()


data33=np.load('X_0_features.npz')
print('hi')
X_0_1=data33['X_0_1']
X_0_2=data33['X_0_2']
X_0_3=data33['X_0_3']
X_0_4=data33['X_0_4']
X_0_5=data33['X_0_5']
X_0_6=data33['X_0_6']
X_0_7=data33['X_0_7']
X_0_8=data33['X_0_8']
X_0_9=data33['X_0_9']
X_0_10=data33['X_0_10']
X_0_11=data33['X_0_11']
X_0_12=data33['X_0_12']
X_0_13=data33['X_0_13']
X_0_14=data33['X_0_14']
X_0_15=data33['X_0_15']
X_0_16=data33['X_0_16']
X_0_17=data33['X_0_17']
X_0_18=data33['X_0_18']
X_0_19=data33['X_0_19']
X_0_20=data33['X_0_20']
X_0_21=data33['X_0_21']
X_0_22=data33['X_0_22']
X_0_23=data33['X_0_23']
X_0_24=data33['X_0_24']
X_0_25=data33['X_0_25']
X_0_26=data33['X_0_26']

X_0 = [X_0_1, X_0_2, X_0_3, X_0_4, X_0_5, X_0_6, X_0_7, X_0_8, X_0_9, X_0_10, X_0_11, X_0_12, X_0_13, X_0_14, X_0_15, X_0_16, X_0_17, X_0_18, X_0_19, X_0_20, X_0_21, X_0_22, X_0_23, X_0_24, X_0_25, X_0_26]

fig = plt.figure()
aa=[0,1,2,8,24,25]
bb=[1,2,10,38,18,27,34,5,7,12,15,29,20,23]
for i in range(14):
    a=fig.add_subplot(7,14, i+1, xticklabels='', yticklabels='')
    plt.imshow(TOT_000[bb[i]*22+0,:,:,:])
    for j in range(6):
        uu = np.array(X_0[aa[j]][bb[i], 0, :])
        a = fig.add_subplot(7, 14, (j+1)*14+i+1, xticklabels='', yticklabels='')
        plt.imshow(np.mean(uu, axis=0))

plt.show()

fig = plt.figure()
aa=[0,1,2,8,24,25]
bb=[1,2,10,38,18,27,34,5,7,12,15,29,20,23]
for i in range(14):
    a=fig.add_subplot(7,14, i+1, xticklabels='', yticklabels='')
    plt.imshow(TOT_000[bb[i]*22+21,:,:,:])
    for j in range(6):
        uu = np.array(X_0[aa[j]][bb[i], 21, :])
        a = fig.add_subplot(7, 14, (j+1)*14+i+1, xticklabels='', yticklabels='')
        plt.imshow(np.mean(uu, axis=0))
plt.show()


exit()



fig = plt.figure()
bb=[1,2,10,38,18,27,34,5,7,12,15,29,20,23]
for i in range(14):
    a=fig.add_subplot(14,27, i*27+1, xticklabels='', yticklabels='')
    plt.imshow(TOT_000[bb[i]*22+11,:,:,:]) 
    for j in range(26):
        uu = np.array(X_0[j][bb[i], 12, :])
        a = fig.add_subplot(14, 27, i*27+j+2, xticklabels='', yticklabels='')
        plt.imshow(np.mean(uu, axis=0))

plt.show()
exit()


fig = plt.figure()


for ii in range (379)
    a=fig.add_subplot(14,27,ii, xticklabels='', yticklabels='')
    uu=np.array(X_0_1[1, 12,:])
    print(uu.shape)
    plt.imshow(np.mean(uu,axis=0))


plt.axis('off')

plt.show()
    

   


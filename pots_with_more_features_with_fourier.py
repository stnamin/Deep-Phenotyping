##### This code is developed to prepare pots data from dataset + handcrafted features (including fourier features) using Grabcut_segmentation_more_features_and_fourier.py  ####

import argparse
import csv
import datetime
import inspect
import json
import logging
import multiprocessing
import os
import re
import shutil
import sys
from time import strptime, strftime, mktime, localtime, struct_time, time, sleep
import warnings
import struct
# Module imports
#import pexif
#import exifread
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import glob

Fourier = np.load('BVZ0072_BVZ0073_97pots_fourier_dictionary.npz')
FOURIER_X_real_dic = Fourier['FOURIER_X_real'].item()
FOURIER_X_imag_dic = Fourier['FOURIER_X_imag'].item()
FOURIER_Y_real_dic = Fourier['FOURIER_Y_real'].item()
FOURIER_Y_imag_dic = Fourier['FOURIER_Y_imag'].item()

No_plants = 97
No_frames = 22
ind = 0
plant_ind = 0        
TOT_X = 300
TOT_Y = 300
TOT_0 = np.zeros((No_plants*No_frames,TOT_X,TOT_Y,3),dtype=np.uint8)
TOT_90 = np.zeros((No_plants*No_frames,TOT_X,TOT_Y,3),dtype=np.uint8)
TOT_180 = np.zeros((No_plants*No_frames,TOT_X,TOT_Y,3),dtype=np.uint8)
TOT_270 = np.zeros((No_plants*No_frames,TOT_X,TOT_Y,3),dtype=np.uint8)
TOT_0_seg = np.zeros((No_plants*No_frames,TOT_X,TOT_Y,3),dtype=np.uint8)
TOT_90_seg = np.zeros((No_plants*No_frames,TOT_X,TOT_Y,3),dtype=np.uint8)
TOT_180_seg = np.zeros((No_plants*No_frames,TOT_X,TOT_Y,3),dtype=np.uint8)
TOT_270_seg = np.zeros((No_plants*No_frames,TOT_X,TOT_Y,3),dtype=np.uint8)
Day = np.zeros(No_plants*No_frames,dtype=np.uint8)

FOURIER_X_real = np.zeros((No_plants*No_frames,255),dtype=np.float64)
FOURIER_X_imag = np.zeros((No_plants*No_frames,255),dtype=np.float64)
FOURIER_Y_real = np.zeros((No_plants*No_frames,255),dtype=np.float64)
FOURIER_Y_imag = np.zeros((No_plants*No_frames,255),dtype=np.float64)


Area = np.zeros(No_plants*No_frames,dtype=np.uint16)
Compactness = np.zeros(No_plants*No_frames,dtype=np.float16)
Eccentricity = np.zeros(No_plants*No_frames,dtype=np.float16)
Extent = np.zeros(No_plants*No_frames,dtype=np.float16)
Max_B = np.zeros(No_plants*No_frames,dtype=np.uint16)
Max_G = np.zeros(No_plants*No_frames,dtype=np.uint16)
Max_R = np.zeros(No_plants*No_frames,dtype=np.uint16)
Min_B = np.zeros(No_plants*No_frames,dtype=np.uint16)
Min_G = np.zeros(No_plants*No_frames,dtype=np.uint16)
Min_R = np.zeros(No_plants*No_frames,dtype=np.uint16)
Mean_B = np.zeros(No_plants*No_frames,dtype=np.uint16)
Mean_G = np.zeros(No_plants*No_frames,dtype=np.uint16)
Mean_R = np.zeros(No_plants*No_frames,dtype=np.uint16)
Mean_H = np.zeros(No_plants*No_frames,dtype=np.uint16)
Mean_V = np.zeros(No_plants*No_frames,dtype=np.uint16)
Mean_S = np.zeros(No_plants*No_frames,dtype=np.uint16)
Perimeter = np.zeros(No_plants*No_frames,dtype=np.uint16)
Roundness = np.zeros(No_plants*No_frames,dtype=np.float16)

Plant_label = np.zeros(No_plants*No_frames,dtype=np.uint8)
Plant_label_single = np.zeros(No_plants,dtype=np.uint8)
Tray_label = np.zeros(No_plants*No_frames,dtype=np.uint8)
Tray_label_single = np.zeros(No_plants,dtype=np.uint8)
BVZ_label = np.zeros(No_plants*No_frames,dtype=np.uint8)
BVZ_label_single = np.zeros(No_plants,dtype=np.uint8)


def pad_str(x):
    """Pads a numeric string to two digits."""
    if len(str(x)) == 1:
        return '0' + str(x)
    return str(x)

with open('./CSVresults/area.csv') as f_area, open('./CSVresults/perimeter.csv') as f_perimeter, open('./CSVresults/roundness.csv') as f_roundness, open('./CSVresults/compactness.csv') as f_compactness, open('./CSVresults/extent.csv') as f_extent, open('./CSVresults/eccentricity.csv') as f_eccentricity, open('./CSVresults/mean_R.csv') as f_R, open('./CSVresults/mean_G.csv') as f_G, open('./CSVresults/mean_B.csv') as f_B, open('./CSVresults/min_R.csv') as f_minR, open('./CSVresults/min_G.csv') as f_minG, open('./CSVresults/min_B.csv') as f_minB, open('./CSVresults/max_R.csv') as f_maxR, open('./CSVresults/max_G.csv') as f_maxG, open('./CSVresults/max_B.csv') as f_maxB, open('./CSVresults/mean_H.csv') as f_H, open('./CSVresults/mean_S.csv') as f_S, open('./CSVresults/mean_V.csv') as f_V:
    area_list = list(csv.reader(f_area))
    perimeter_list = list(csv.reader(f_perimeter))
    roundness_list = list(csv.reader(f_roundness))
    compactness_list = list(csv.reader(f_compactness))
    extent_list = list(csv.reader(f_extent))
    eccentricity_list = list(csv.reader(f_eccentricity))
    R_list = list(csv.reader(f_R))
    G_list = list(csv.reader(f_G))
    B_list = list(csv.reader(f_B))
    Rmin_list = list(csv.reader(f_minR))
    Gmin_list = list(csv.reader(f_minG))
    Bmin_list = list(csv.reader(f_minB))
    Rmax_list = list(csv.reader(f_maxR))
    Gmax_list = list(csv.reader(f_maxG))
    Bmax_list = list(csv.reader(f_maxB))
    H_list = list(csv.reader(f_H))
    S_list = list(csv.reader(f_S))
    V_list = list(csv.reader(f_V))
    
    
    
for dirpath, dirnames, filenames in os.walk('./BVZ0072_BVZ0073'):
    for dir in dirnames:
        Plant_label_single[plant_ind] = int(dir[-7])
        Tray_label_single[plant_ind] = int(dir[-11:-9])
        BVZ_label_single[plant_ind] = int(dir[5:7])
        plant_ind = plant_ind + 1
        print(plant_ind)
        print(dir, dir[5:7],dir[-7],dir[-11:-9])
        ind2 = 0
        with open('area.csv') as f:
            area_list = csv.reader(f)
            #print(area_list[0])
            for row in area_list:
                if row[0] == dir:
                    break
                else:
                    ind2 = ind2 + 1
        for day in np.arange(1,23):
            a = glob.glob('./BVZ0072_BVZ0073/'+dir+'/'+dir+'_2016_10_'+pad_str(day)+'*.jpg')
            b = glob.glob(('./BVZ0072_BVZ0073_seg/'+dir+'/'+dir+'_2016_10_'+pad_str(day)+'*.jpg').replace('cor','seg'))

            filename = a[0]
            print(os.path.basename(filename))
            Plant_label[ind] = int(a[0][55])
            Tray_label[ind] = int(a[0][51:53])
            BVZ_label[ind] = int(a[0][68:70])
            Area[ind] = np.uint16(float(row[day]))
            Compactness[ind] = float(float(compactness_list[ind2][day]))
            Eccentricity[ind] = float(float(eccentricity_list[ind2][day]))
            Extent[ind] = float(float(extent_list[ind2][day]))
            Max_B[ind] = np.uint16(float(Bmax_list[ind2][day]))
            Max_G[ind] = np.uint16(float(Gmax_list[ind2][day]))
            Max_R[ind] = np.uint16(float(Rmax_list[ind2][day]))
            Min_B[ind] = np.uint16(float(Bmin_list[ind2][day]))
            Min_G[ind] = np.uint16(float(Gmin_list[ind2][day]))
            Min_R[ind] = np.uint16(float(Rmin_list[ind2][day]))
            Mean_B[ind] = np.uint16(float(B_list[ind2][day]))
            Mean_G[ind] = np.uint16(float(G_list[ind2][day]))
            Mean_R[ind] = np.uint16(float(R_list[ind2][day]))
            Mean_H[ind] = np.uint16(float(H_list[ind2][day]))
            Mean_S[ind] = np.uint16(float(S_list[ind2][day]))
            Mean_V[ind] = np.uint16(float(V_list[ind2][day]))
            Perimeter[ind] = np.uint16(float(perimeter_list[ind2][day]))
            Roundness[ind] = float(roundness_list[ind2][day])
            
            FOURIER_X_real[ind,:] = FOURIER_X_real_dic[os.path.basename(filename)]
            FOURIER_X_imag[ind,:] = FOURIER_X_imag_dic[os.path.basename(filename)]
            FOURIER_Y_real[ind,:] = FOURIER_Y_real_dic[os.path.basename(filename)]
            FOURIER_Y_imag[ind,:] = FOURIER_Y_imag_dic[os.path.basename(filename)]
            
            #print(a[0],ind,int(a[0][55]),int(a[0][51:53]),int(a[0][68:70]))
            '''
            tempp3 = cv2.imread(filename)
            tempp2 = tempp3[19:-1,10:310,:]
            print(b[0])
            tempp4 = cv2.imread(b[0])
            TOT_0[ind,:,:,:] = tempp2
            tempp2 = np.rot90(tempp2)
            TOT_90[ind,:,:,:] = tempp2
            tempp2 = np.rot90(tempp2)
            TOT_180[ind,:,:,:] = tempp2
            tempp2 = np.rot90(tempp2)
            TOT_270[ind,:,:,:] = tempp2
            TOT_0_seg[ind,:,:,:] = tempp4
            tempp4 = np.rot90(tempp4)
            TOT_90_seg[ind,:,:,:] = tempp4
            tempp4 = np.rot90(tempp4)
            TOT_180_seg[ind,:,:,:] = tempp4
            tempp4 = np.rot90(tempp4)
            TOT_270_seg[ind,:,:,:] = tempp4
            '''
            
            ind = ind + 1
        #print(Area[ind-1])
        #print(Roundness[ind-3:ind])
        #print(Compactness[ind-3:ind])

#print(Tray_label)
print(FOURIER_X_imag.shape)
#print(Tray_label_single)
#print(Area)
#print(sum(Plant_label_single==1),sum(Plant_label_single==2),sum(Plant_label_single==3),sum(Plant_label_single==4))
#print(sum(BVZ_label_single==72), sum(BVZ_label_single==73))
#np.savez('BVZ0072_BVZ0073_97pots', TOT_0=TOT_0, TOT_90=TOT_90, TOT_180=TOT_180, TOT_270=TOT_270, Plant_label=Plant_label, Plant_label_single=Plant_label_single, Tray_label = Tray_label, Tray_label_single=Tray_label_single, BVZ_label = BVZ_label, BVZ_label_single=BVZ_label_single, TOT_0_seg=TOT_0_seg, TOT_90_seg=TOT_90_seg, TOT_180_seg=TOT_180_seg, TOT_270_seg=TOT_270_seg, Area = Area)
#np.savez('BVZ0072_BVZ0073_97pots_features', Area = Area, Compactness = Compactness, Eccentricity = Eccentricity, Extent = Extent, Max_B = Max_B, Max_G = Max_G, Max_R = Max_R, Min_B = Min_B, Min_G = Min_G, Min_R = Min_R, Mean_B = Mean_B, Mean_G = Mean_G, Mean_R = Mean_R, Mean_H = Mean_H, Mean_S = Mean_S, Mean_V = Mean_V, Perimeter = Perimeter, Roundness = Roundness)
np.savez('BVZ0072_BVZ0073_97pots_fourier_features', FOURIER_X_real = FOURIER_X_real, FOURIER_X_imag = FOURIER_X_imag, FOURIER_Y_imag = FOURIER_Y_imag, FOURIER_Y_real = FOURIER_Y_real)
#np.savez('BVZ0072_BVZ0073_108pots', TOT_0=TOT_0)                

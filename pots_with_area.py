##### This code is developed to prepare pots data + area from dataset ####

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
Area = np.zeros(No_plants*No_frames,dtype=np.uint16)
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

for dirpath, dirnames, filenames in os.walk('./BVZ0072_BVZ0073'):
    for dir in dirnames:
        Plant_label_single[plant_ind] = int(dir[-7])
        Tray_label_single[plant_ind] = int(dir[-11:-9])
        BVZ_label_single[plant_ind] = int(dir[5:7])
        plant_ind = plant_ind + 1
        print(plant_ind)
        print(dir, dir[5:7],dir[-7],dir[-11:-9])
        with open('area.csv') as f:
            area_list = csv.reader(f)
            #print(area_list[0])
            for row in area_list:
                if row[0] == dir:
                    break
                
        for day in np.arange(1,23):
            a = glob.glob('./BVZ0072_BVZ0073/'+dir+'/'+dir+'_2016_10_'+pad_str(day)+'*.jpg')
            b = glob.glob(('./BVZ0072_BVZ0073_seg/'+dir+'/'+dir+'_2016_10_'+pad_str(day)+'*.jpg').replace('cor','seg'))

            filename = a[0]
            Plant_label[ind] = int(a[0][55])
            Tray_label[ind] = int(a[0][51:53])
            BVZ_label[ind] = int(a[0][68:70])
            Area[ind] = np.uint16(float(row[day]))
            #print(a[0],ind,int(a[0][55]),int(a[0][51:53]),int(a[0][68:70]))
            tempp3 = cv2.imread(filename)
            tempp2 = tempp3[19:-1,10:310,:]
            print(b[0])
            tempp4 = cv2.imread(b[0])
            #print(tempp2.shape)
            TOT_0[ind,:,:,:] = tempp2
            tempp2 = np.rot90(tempp2)
            TOT_90[ind,:,:,:] = tempp2
            tempp2 = np.rot90(tempp2)
            TOT_180[ind,:,:,:] = tempp2
            tempp2 = np.rot90(tempp2)
            TOT_270[ind,:,:,:] = tempp2
            #tempp2 = np.rot90(tempp2)
            TOT_0_seg[ind,:,:,:] = tempp4
            tempp4 = np.rot90(tempp4)
            TOT_90_seg[ind,:,:,:] = tempp4
            tempp4 = np.rot90(tempp4)
            TOT_180_seg[ind,:,:,:] = tempp4
            tempp4 = np.rot90(tempp4)
            TOT_270_seg[ind,:,:,:] = tempp4
            
            
            ind = ind + 1
        print(Area[ind-1])
print(Tray_label)
print(TOT_0.shape)
print(Tray_label_single)
print(Area)
print(sum(Plant_label_single==1),sum(Plant_label_single==2),sum(Plant_label_single==3),sum(Plant_label_single==4))
print(sum(BVZ_label_single==72), sum(BVZ_label_single==73))
np.savez('BVZ0072_BVZ0073_97pots_with_area', TOT_0=TOT_0, TOT_90=TOT_90, TOT_180=TOT_180, TOT_270=TOT_270, Plant_label=Plant_label, Plant_label_single=Plant_label_single, Tray_label = Tray_label, Tray_label_single=Tray_label_single, BVZ_label = BVZ_label, BVZ_label_single=BVZ_label_single, TOT_0_seg=TOT_0_seg, TOT_90_seg=TOT_90_seg, TOT_180_seg=TOT_180_seg, TOT_270_seg=TOT_270_seg, Area = Area)
#np.savez('BVZ0072_BVZ0073_108pots', TOT_0=TOT_0)                

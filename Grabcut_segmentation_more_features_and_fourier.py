##### This code is developed for segmentation and handcrafted features extraction  ####

import numpy as np
import cv2
import sys
import glob
from matplotlib import pyplot as plt
import csv
import os


rootdir = '/network/largedata/Storage/users/Mohammad/BVZ0072_BV0073_dataset/BVZ0072_BVZ0073'
All_folders = [ name for name in os.listdir(rootdir) if os.path.isdir(os.path.join(rootdir, name))]

mask3 = cv2.imread('Mask_BVZ0072.jpg',0)
kernel = np.ones((7,7),np.uint8)

ret, mask3 = cv2.threshold(mask3, 127, 1, cv2.THRESH_BINARY)

HSVGreenMin=np.array([20, 40, 40])
HSVGreenMax=np.array([100, 255, 180])

HSVMin=np.array([10, 30, 40])
HSVMax=np.array([110, 255, 180])

with open('./CSVresults/area.csv', 'wb') as f_area, open('./CSVresults/perimeter.csv', 'wb') as f_perimeter, open('./CSVresults/roundness.csv', 'wb') as f_roundness, open('./CSVresults/compactness.csv', 'wb') as f_compactness, open('./CSVresults/extent.csv', 'wb') as f_extent, open('./CSVresults/eccentricity.csv', 'wb') as f_eccentricity, open('./CSVresults/mean_R.csv', 'wb') as f_R, open('./CSVresults/mean_G.csv', 'wb') as f_G, open('./CSVresults/mean_B.csv', 'wb') as f_B, open('./CSVresults/min_R.csv', 'wb') as f_minR, open('./CSVresults/min_G.csv', 'wb') as f_minG, open('./CSVresults/min_B.csv', 'wb') as f_minB, open('./CSVresults/max_R.csv', 'wb') as f_maxR, open('./CSVresults/max_G.csv', 'wb') as f_maxG, open('./CSVresults/max_B.csv', 'wb') as f_maxB, open('./CSVresults/mean_H.csv', 'wb') as f_H, open('./CSVresults/mean_S.csv', 'wb') as f_S, open('./CSVresults/mean_V.csv', 'wb') as f_V:
    writer_area = csv.writer(f_area)
    writer_perimeter = csv.writer(f_perimeter)
    writer_roundness = csv.writer(f_roundness)
    writer_compactness = csv.writer(f_compactness)
    writer_extent = csv.writer(f_extent)
    writer_eccentricity = csv.writer(f_eccentricity)    
    writer_R = csv.writer(f_R)
    writer_G = csv.writer(f_G)
    writer_B = csv.writer(f_B)
    writer_Rmin = csv.writer(f_minR)
    writer_Gmin = csv.writer(f_minG)
    writer_Bmin = csv.writer(f_minB)
    writer_Rmax = csv.writer(f_maxR)
    writer_Gmax = csv.writer(f_maxG)
    writer_Bmax = csv.writer(f_maxB)
    writer_H = csv.writer(f_H)
    writer_S = csv.writer(f_S)
    writer_V = csv.writer(f_V)
    
    Audit_fields = (('Name','Area1','Area2','Area3','Area4','Area5','Area6','Area7','Area8','Area9','Area10','Area11','Area12','Area13','Area14','Area15','Area16','Area17','Area18','Area19','Area20','Area21','Area22'))
    writer_area.writerow( Audit_fields )
    writer_perimeter.writerow( Audit_fields )
    writer_roundness.writerow( Audit_fields )
    writer_compactness.writerow( Audit_fields )
    writer_extent.writerow( Audit_fields )
    writer_eccentricity.writerow( Audit_fields )
    writer_R.writerow( Audit_fields )
    writer_G.writerow( Audit_fields )
    writer_B.writerow( Audit_fields )
    writer_Rmin.writerow( Audit_fields )
    writer_Gmin.writerow( Audit_fields )
    writer_Bmin.writerow( Audit_fields )
    writer_Rmax.writerow( Audit_fields )
    writer_Gmax.writerow( Audit_fields )
    writer_Bmax.writerow( Audit_fields )
    writer_H.writerow( Audit_fields )
    writer_S.writerow( Audit_fields )
    writer_V.writerow( Audit_fields )
    FOURIER_X_real = {}
    FOURIER_X_imag = {}
    FOURIER_Y_real = {}
    FOURIER_Y_imag = {}
    for DIR in All_folders:
        DIR_tot = rootdir + '/' + DIR
        if not os.path.exists((DIR_tot.replace('/BVZ0072_BVZ0073/','/BVZ0072_BVZ0073_contour/')).replace('-cor','-con')):
            os.makedirs((DIR_tot.replace('/BVZ0072_BVZ0073/','/BVZ0072_BVZ0073_contour/')).replace('-cor','-con'))
        if not os.path.exists((DIR_tot.replace('/BVZ0072_BVZ0073/','/BVZ0072_BVZ0073_seg/')).replace('-cor','-seg')):
            os.makedirs((DIR_tot.replace('/BVZ0072_BVZ0073/','/BVZ0072_BVZ0073_seg/')).replace('-cor','-seg'))
        a = sorted(glob.glob(DIR_tot+'/*.jpg'))
        Area = (DIR,)
        Perimeter = (DIR,)
        Roundness = (DIR,)
        Compactness = (DIR,)
        Extent = (DIR,)
        Eccentricity = (DIR,)

        MEAN_R = (DIR,)
        MEAN_G = (DIR,)
        MEAN_B = (DIR,)
        MIN_R = (DIR,)
        MIN_G = (DIR,)
        MIN_B = (DIR,)
        MAX_R = (DIR,)
        MAX_G = (DIR,)
        MAX_B = (DIR,)
        MEAN_H = (DIR,)
        MEAN_S = (DIR,)
        MEAN_V = (DIR,)

        for i in a:
            print(i)


            tempp = cv2.imread(i)
            img = tempp[19:-1,10:310,:]
            IND = np.where(mask3==0)
            img[IND[0],IND[1],:] = [255,255,255]
            
            HSVImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            LabImage = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            
            
            Thresh2 = np.uint8((cv2.inRange(HSVImage, HSVMin, HSVMax))>0)
            rect = (0,0,299,299)

            blur = LabImage[:,:,2]#cv2.GaussianBlur(LabImage[:,:,2],(3,3),0)#
            blur2 = LabImage[:,:,1]#cv2.GaussianBlur(LabImage[:,:,1],(3,3),0)#
            ret2, otsu2 = cv2.threshold(blur2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            otsu = blur > 151

            otsu = np.uint8((otsu>0)*np.uint8(Thresh2))
            otsu_dilated = cv2.dilate(otsu,kernel,iterations = 2)
            otsu = (3*(otsu + otsu_dilated))%5
            otsu2 = np.uint8((otsu2==0)*np.uint8(Thresh2))
            otsu2_dilated = cv2.dilate(otsu2,kernel,iterations = 2)
            otsu2 = (3*(otsu2 + otsu2_dilated))%5


            img2 = img.copy()

            #HSVImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            Thresh = np.uint8((cv2.inRange(HSVImage, HSVGreenMin, HSVGreenMax)))>0
            IND = np.where(Thresh>0)
            rect = (np.max((0,np.min(IND[1])-5)),np.max((0,np.min(IND[0])-5)),np.min((299,np.max(IND[1])+5)),np.min((299,np.max(IND[0])+5)))

            mask = np.zeros(img.shape[:2],dtype = np.uint8) 


            mask_temp = otsu+otsu2
            mask = np.uint8(mask_temp==2) + 3*np.uint8(mask_temp==6) + 3*np.uint8(mask_temp==1) + 3*np.uint8(mask_temp==4) + 2*np.uint8(mask_temp==3)

            bgdmodel = np.zeros((1,65),np.float64)
            fgdmodel = np.zeros((1,65),np.float64)
            cv2.grabCut(img2,mask,rect,bgdmodel,fgdmodel,6,cv2.GC_INIT_WITH_MASK)
            mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')

            kernel = np.ones((5,5),np.uint8)
            kernel2 = np.ones((5,5),np.uint8)

            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)

            mask_analyzed = cv2.connectedComponentsWithStats(mask2, 4, cv2.CV_32S)

            IND = ((mask_analyzed[2][1:,4]<70000) & (mask_analyzed[3][1:,0]>80) & (mask_analyzed[3][1:,0]<220) & (mask_analyzed[3][1:,1]>80) & (mask_analyzed[3][1:,1]<220) & ((np.abs(mask_analyzed[2][1:,4]/np.float32(mask_analyzed[2][1:,2]*mask_analyzed[2][1:,3])))>.3))

            test = mask_analyzed[2][1:,4]
            test2 = np.argmax(test[IND])
            final_mask = np.uint8(mask_analyzed[1]==np.where(IND==True)[0][test2]+1)

            im2, cnt, hierarchy = cv2.findContours(final_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            x = cnt[0][:][:,0]
            x_fft = np.fft.fft2(x)
            x_fft[0] = [0,0] # to make the centre on (0,0)
            x_zero_centroid = np.fft.ifft2(x_fft)
            x_fft_2 = np.fft.fft2(x_zero_centroid.real,[512,2])
            FOURIER_X_real[os.path.basename(i)] =  x_fft_2[1:256,0].real
            FOURIER_X_imag[os.path.basename(i)] =  x_fft_2[1:256,0].imag
            FOURIER_Y_real[os.path.basename(i)] =  x_fft_2[1:256,1].real
            FOURIER_Y_imag[os.path.basename(i)] =  x_fft_2[1:256,1].imag
            

            IND = np.where(final_mask>0)
            
            MEAN_R2 = np.mean(img2[IND[0],IND[1],2])
            MEAN_R = MEAN_R + (MEAN_R2,)
            MEAN_G2 = np.mean(img2[IND[0],IND[1],1])
            MEAN_G = MEAN_G + (MEAN_G2,)
            MEAN_B2 = np.mean(img2[IND[0],IND[1],0])
            MEAN_B = MEAN_B + (MEAN_B2,)
            
            MIN_R2 = np.min(img2[IND[0],IND[1],2])
            MIN_R = MIN_R + (MIN_R2,)
            MIN_G2 = np.min(img2[IND[0],IND[1],1])
            MIN_G = MIN_G + (MIN_G2,)
            MIN_B2 = np.min(img2[IND[0],IND[1],0])
            MIN_B = MIN_B + (MIN_B2,)
            
            MAX_R2 = np.max(img2[IND[0],IND[1],2])
            MAX_R = MAX_R + (MAX_R2,)
            MAX_G2 = np.max(img2[IND[0],IND[1],1])
            MAX_G = MAX_G + (MAX_G2,)
            MAX_B2 = np.max(img2[IND[0],IND[1],0])
            MAX_B = MAX_B + (MAX_B2,)
            
            #Mean_G = np.mean(img2[IND[0],IND[1],1])
            #Mean_B = np.mean(img2[IND[0],IND[1],0])
            MEAN_H2 = np.mean(HSVImage[IND[0],IND[1],0])
            MEAN_H = MEAN_H + (MEAN_H2,)
            MEAN_S2 = np.mean(HSVImage[IND[0],IND[1],1])
            MEAN_S = MEAN_S + (MEAN_S2,)
            MEAN_V2 = np.mean(HSVImage[IND[0],IND[1],2])
            MEAN_V = MEAN_V + (MEAN_V2,)
            
            Area2 = cv2.contourArea(cnt[0])
            Area = Area + (Area2,)
            
            Perimeter2 = cv2.arcLength(cnt[0],True)
            Perimeter = Perimeter + (Perimeter2,)
            
            Roundness2 = Area2/float(Perimeter2)
            Roundness = Roundness + (Roundness2,)
            
            hull = cv2.convexHull(cnt[0])
            hull_area = cv2.contourArea(hull)

            Compactness2 = Area2/float(hull_area)
            Compactness = Compactness + (Compactness2,)
            
            
            x,y,w,h = cv2.boundingRect(cnt[0])
            rect_area = w*h
            Extent2 = float(Area2)/rect_area
            Extent = Extent + (Extent2,)
            
            
            (x,y),(MA,ma),angle = cv2.fitEllipse(cnt[0])
            if MA > ma:
                Eccentricity2 = ma/float(MA)
            else:
                Eccentricity2 = MA/float(ma)
            
            Eccentricity = Eccentricity + (Eccentricity2,)

            output = cv2.bitwise_and(img,img,mask=final_mask)

            cv2.drawContours(img,cnt,-1,(255,255,255),1)

            cv2.imwrite((i.replace('/BVZ0072_BVZ0073/','/BVZ0072_BVZ0073_contour/')).replace('-cor','-con'),img)
            cv2.imwrite((i.replace('/BVZ0072_BVZ0073/','/BVZ0072_BVZ0073_seg/')).replace('-cor','-seg'),output)
        print(Area)
        writer_area.writerow( Area )
        writer_perimeter.writerow( Perimeter )
        writer_roundness.writerow( Roundness )
        writer_compactness.writerow( Compactness )
        writer_extent.writerow( Extent )
        writer_eccentricity.writerow( Eccentricity )
        
        writer_R.writerow( MEAN_R )
        writer_G.writerow( MEAN_G )
        writer_B.writerow( MEAN_B )
        writer_Rmin.writerow( MIN_R )
        writer_Gmin.writerow( MIN_G )
        writer_Bmin.writerow( MIN_B )
        writer_Rmax.writerow( MAX_R )
        writer_Gmax.writerow( MAX_G )
        writer_Bmax.writerow( MAX_B )
        writer_H.writerow( MEAN_H )
        writer_S.writerow( MEAN_S )
        writer_V.writerow( MEAN_V )
        
np.savez('BVZ0072_BVZ0073_97pots_fourier_dictionary', FOURIER_X_real = FOURIER_X_real, FOURIER_X_imag = FOURIER_X_imag, FOURIER_Y_imag = FOURIER_Y_imag, FOURIER_Y_real = FOURIER_Y_real)



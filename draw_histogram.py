# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 15:44:11 2019

@author: Administrator
"""
import os
import cv2
from scipy import misc
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
plt.figure()#新建一个图像
for ind in range(0,1): #9为测试集
    path1 = os.listdir(r"E:\data\after_spm\IMAGES_OF_GE\%d"%ind)
    path1.sort(key=lambda x:int(x[:-4]))
    path2 = os.listdir(r"E:\data\after_spm\IMAGES_OF_XI\%d"%ind)
    path2.sort(key=lambda x:int(x[:-4]))
    for filename,filename_1 in zip(path1,path2):  
        img=io.imread('E:/data/after_spm/IMAGES_OF_GE/%d/%s'%(ind,filename))
        img_1=io.imread('E:/data/after_spm/IMAGES_OF_XI/%d/%s'%(ind,filename_1))
        tupian1 = io.imread('E:/data/after_spm/IMAGES_OF_GE/0/17.jpg')
        tupian2 = io.imread('E:/data/after_spm/IMAGES_OF_XI/0/17.jpg')
        hist = cv2.calcHist([img],[0],None,[256],[0,256])
        hist_1 = cv2.calcHist([img_1],[0],None,[256],[0,256])
        plt.subplot(221),plt.imshow(tupian1)
        plt.title("Origin Images of GE")#图像的标题
        plt.subplot(222),plt.imshow(tupian2)
        plt.title("Origin Images of Siemens")#图像的标题
        plt.subplot(212),plt.plot(hist,color='red')#画图
        plt.plot(hist_1,color='blue')#画图
        
        plt.xlabel("Intensity")#X轴标签
        plt.ylabel("Number of Pixels")#Y轴标签
        plt.xlim([0,256])#设置x坐标轴范围
plt.show()#显示图像


plt.figure()#新建一个图像
for ind in range(1,22): #9为测试集
        img=io.imread('E:\\data\\after_spm\\statistical-standardisation\\mean\\%d.jpg'%(ind))
        img_1=io.imread('E:\\data\\after_spm\\statistical-standardisation_1\\mean\\%d.jpg'%(ind))
        tupian1 = io.imread('E:/data/after_spm/statistical-standardisation/mean/17.jpg')
        tupian2 = io.imread('E:/data/after_spm/statistical-standardisation_1/mean/17.jpg')
        hist = cv2.calcHist([img],[0],None,[256],[-68,512])
        hist_1 = cv2.calcHist([img_1],[0],None,[256],[-68,512])

        plt.subplot(221),plt.imshow(tupian1)
        plt.title("After standardisation using mean(GE)")#图像的标题
        plt.subplot(222),plt.imshow(tupian2)
        plt.title("After standardisation using mean(Siemens)")#图像的标题
        plt.subplot(212),plt.plot(hist,color='red')#画图
        plt.plot(hist_1,color='blue')#画图
        
        plt.xlabel("Intensity")#X轴标签
        plt.ylabel("Number of Pixels")#Y轴标签
        plt.xlim([0,256])#设置x坐标轴范围

plt.show()#显示图像


plt.figure()#新建一个图像
for ind in range(0,1): #9为测试集
    path1 = os.listdir(r"E:\data\after_spm\statistical-standardisation\z-score")
    #path1.sort(key=lambda x:int(x[:-4]))
    path2 = os.listdir(r"E:\data\after_spm\statistical-standardisation_1\z-score")
    #path2.sort(key=lambda x:int(x[:-4]))
    for filename,filename_1 in zip(path1,path2):  
        img=io.imread('E:\\data\\after_spm\\statistical-standardisation\\z-score\\%s'%(filename))
        img_1=io.imread('E:\\data\\after_spm\\statistical-standardisation_1\\z-score\\%s'%(filename_1))
        tupian1 = io.imread('E:/data/after_spm/statistical-standardisation/z-score/0-17.jpg')
        tupian2 = io.imread('E:/data/after_spm/statistical-standardisation_1/z-score/0-17.jpg')
        hist = cv2.calcHist([img],[0],None,[256],[0,256])
        hist_1 = cv2.calcHist([img_1],[0],None,[256],[0,256])
        
        plt.subplot(221),plt.imshow(tupian1)
        plt.title("After standardisation using Z-score(GE)")#图像的标题
        plt.subplot(222),plt.imshow(tupian2)
        plt.title("After standardisation using Z-score(Siemens)")#图像的标题
        plt.subplot(212),plt.plot(hist,color='red')#画图
        plt.plot(hist_1,color='blue')#画图
        
        plt.xlabel("Intensity")#X轴标签
        plt.ylabel("Number of Pixels")#Y轴标签
        plt.xlim([0,256])#设置x坐标轴范围

plt.show()#显示图像

plt.figure()#新建一个图像
for ind in range(0,1): #9为测试集
    path1 = os.listdir(r"E:\data\after_spm\statistical-standardisation\median+IQR")
    #path1.sort(key=lambda x:int(x[:-4]))
    path2 = os.listdir(r"E:\data\after_spm\statistical-standardisation_1\median+IQR")
    #path2.sort(key=lambda x:int(x[:-4]))
    for filename,filename_1 in zip(path1,path2):  
        img=io.imread('E:\\data\\after_spm\\statistical-standardisation\\median+IQR\\%s'%(filename))
        img_1=io.imread('E:\\data\\after_spm\\statistical-standardisation_1\\median+IQR\\%s'%(filename_1))
        tupian1 = io.imread('E:/data/after_spm/statistical-standardisation/median+IQR/0-17.jpg')
        tupian2 = io.imread('E:/data/after_spm/statistical-standardisation_1/median+IQR/0-17.jpg')
        hist = cv2.calcHist([img],[0],None,[256],[0,256])
        hist_1 = cv2.calcHist([img_1],[0],None,[256],[0,256])
        
        plt.subplot(221),plt.imshow(tupian1)
        plt.title("After standardisation using median+IQR(Siemens)")#图像的标题
        plt.subplot(222),plt.imshow(tupian2)
        plt.title("After standardisation using median+IQR(Siemens)")#图像的标题
        plt.subplot(212),plt.plot(hist,color='red')#画图
        plt.plot(hist_1,color='blue')#画图
        
        plt.xlabel("Intensity")#X轴标签
        plt.ylabel("Number of Pixels")#Y轴标签
        plt.xlim([0,256])#设置x坐标轴范围

plt.show()#显示图像



        
                
            
            
            

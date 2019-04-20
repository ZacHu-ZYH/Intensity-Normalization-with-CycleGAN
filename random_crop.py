# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 21:59:48 2019

@author: Administrator
"""
import numpy as np
import os
from scipy import misc
#size 436*363
import cv2
import tensorflow as tf
import random

for ind in range(0,20): #9为测试集
    path1 = os.listdir(r"E:\data\after_spm\IMAGES_OF_GE\%d"%ind)
    path1.sort(key=lambda x:int(x[:-4]))
    path2 = os.listdir(r"E:\data\after_spm\IMAGES_OF_XI\%d"%ind)
    path2.sort(key=lambda x:int(x[:-4]))
    for filename,filename_1 in zip(path1,path2):  
     
        img = cv2.imread(r"E:\data\after_spm\IMAGES_OF_GE\%d\%s"%(ind,filename))
        img_1 = cv2.imread(r"E:\data\after_spm\IMAGES_OF_XI\%d\%s"%(ind,filename_1))
        
        sess	=	tf.InteractiveSession()	

        
        #h、w为想要截取的图片大小
        h=256
        w=256
         
        count=0
        while 1:
        
            y = random.randint(1, 436-w)
            x = random.randint(1, 363-h)
            
            cropImg = img[(y):(y + h), (x):(x + w)]
            cropImg_1 = img_1[(y):(y + h), (x):(x + w)]
            cv2.imwrite("E:\\data\\after_spm\\after_random_crop\\%d-%s-%d.jpg"%(ind,filename,count), cropImg)
            cv2.imwrite("E:\\data\\after_spm\\after_random_crop1\\%d-%s-%d.jpg"%(ind,filename_1,count),cropImg_1)
            count+=1
             
            if count==20:
                break
                 
               
        cv2.waitKey(0) 
        sess.close()
        

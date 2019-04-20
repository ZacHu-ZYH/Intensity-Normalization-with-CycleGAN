# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 14:06:27 2019

@author: Administrator
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from skimage import transform,data
from PIL import Image,ImageTk  
import math


def resize( w_box, h_box, pil_image): #参数是：要适应的窗口宽、高、Image.open后的图片
  w, h = pil_image.size #获取图像的原始大小   
  f1 = 1.0*w_box/w 
  f2 = 1.0*h_box/h    
  factor = min([f1, f2])   
  width = int(w*factor)    
  height = int(h*factor)    
  return pil_image.resize((width, height), Image.ANTIALIAS)


def getMatchNum(matches,ratio):
    '''返回特征点匹配数量和匹配掩码'''
    matchesMask=[[0,0] for i in range(len(matches))]
    matchNum=0
    for i,(m,n) in enumerate(matches):
        if m.distance<ratio*n.distance: #将距离比率小于ratio的匹配点删选出来
            matchesMask[i]=[1,0]
            matchNum+=1
    return (matchNum,matchesMask)

#queryPath='F:\\data\\after_spm\\IMAGES OF XI\\10\\'
queryPath='F:\\data\\after_spm\\after_skull_stripping_of_xi_images\\10\\'
#samplePath='F:\\data\\after_spm\\IMAGES OF GE\\0\\1#11.jpg' #样本图片
samplePath='F:\\data\\after_spm\\after_skull_stripping_of_ge_images\\7\\22#12.jpg' #样本图片
comparisonImageList=[] #记录比较结果

#创建SIFT特征提取器
sift = cv2.xfeatures2d.SIFT_create() 
#创建FLANN匹配对象 
FLANN_INDEX_KDTREE=0
indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
searchParams=dict(checks=50)
flann=cv2.FlannBasedMatcher(indexParams,searchParams)

sampleImage=cv2.imread(samplePath,0)
kp1, des1 = sift.detectAndCompute(sampleImage, None) #提取样本图片的特征
for parent,dirnames,filenames in os.walk(queryPath):
    for p in filenames:
        p=queryPath+p
        print(p)
        queryImage=cv2.imread(p,0)
        kp2, des2 = sift.detectAndCompute(queryImage, None) #提取比对图片的特征
        matches=flann.knnMatch(des1,des2,k=2) #匹配特征点，为了删选匹配点，指定k为2，这样对样本图的每个特征点，返回两个匹配
        (matchNum,matchesMask)=getMatchNum(matches,0.93) #通过比率条件，计算出匹配程度
        matchRatio=matchNum*100/len(matches)
        drawParams=dict(matchColor=(96,96,96),
                singlePointColor=(96,96,96),
                matchesMask=matchesMask,
                flags=0)
        comparisonImage=cv2.drawMatchesKnn(sampleImage,kp1,queryImage,kp2,matches,None,**drawParams)
        #p = p[34:60]
        p = p[56:65]
        print(p)
        comparisonImageList.append((comparisonImage,matchRatio,p)) #记录下结果

comparisonImageList.sort(key=lambda x:x[1],reverse=True) #按照匹配度排序
count=len(comparisonImageList)
column=4
row=math.ceil(count/column)


#绘图显示
figure,ax=plt.subplots(row,column)
for index,(image,ratio,p) in enumerate(comparisonImageList):
    ax[int(index/column)][index%column].set_title('%.4f%%' % ratio,loc='left')
    ax[int(index/column)][index%column].set_title('%s' %p,loc='right')
    ax[int(index/column)][index%column].imshow(image)
    
    


plt.show()

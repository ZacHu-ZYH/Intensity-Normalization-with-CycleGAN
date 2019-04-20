# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:36:27 2019

@author: Administrator
"""

#from skimage.measure import compare_ssim as ssimm
#from skimage.measure import compare_psnr as psnrr
from skimage.measure import compare_mse as msee
import numpy as np
import cv2
from scipy.signal import convolve2d
from PIL import Image

np.seterr(divide='ignore',invalid='ignore')

def convolution(img,kernal):
    new_arr = kernal.reshape(kernal.size)
    new_arr = new_arr[::-1]
    kernal = new_arr.reshape(kernal.shape)

    kernal_heigh = kernal.shape[0]
    kernal_width = kernal.shape[1]

    cor_heigh = img.shape[0] - kernal_heigh + 1
    cor_width = img.shape[1] - kernal_width + 1
    result = np.zeros((cor_heigh, cor_width), dtype=np.float64)
    for i in range(cor_heigh):
        for j in range(cor_width):
            result[i][j] = (img[i:i + kernal_heigh, j:j + kernal_width] * kernal).sum()
    return result

def gmsdd(dis_img,ref_img,c=1):
    hx=np.array([[1/3,0,-1/3]]*3,dtype=np.float64)#Prewitt算子
    ave_filter=np.array([[0.25,0.25],[0.25,0.25]])#均值滤波核
    down_step=2#下采样间隔
    hy=hx.transpose()
    #均值滤波
    ave_dis=convolution(dis_img,ave_filter)
    ave_ref=convolution(ref_img,ave_filter)
    #下采样
    ave_dis_down=ave_dis[np.arange(0,ave_dis.shape[0],down_step),:]
    ave_dis_down=ave_dis_down[:,np.arange(0,ave_dis_down.shape[1],down_step)]
    ave_ref_down=ave_ref[np.arange(0,ave_ref.shape[0],down_step),:]
    ave_ref_down=ave_ref_down[:,np.arange(0,ave_ref_down.shape[1],down_step)]
    #计算mr md等中间变量
    mr_sq=convolution(ave_ref_down,hx)**2+convolution(ave_ref_down,hy)**2
    md_sq=convolution(ave_dis_down,hx)**2+convolution(ave_dis_down,hy)**2
    mr=np.sqrt(mr_sq)
    md=np.sqrt(md_sq)
    GMS=(2*mr*md+c)/(mr_sq+md_sq+c)
    GMSM=np.mean(GMS)
    GMSD=np.mean((GMS-GMSM)**2)
    return GMSD*2

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def compute_ssim(im1, im2, k1=0.1, k2=0.3, win_size=1, L=255):

    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2

    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))

    return np.mean(np.mean(ssim_map))
   
list_mse=[]    
list_ssim=[]   
list_psnr=[]    
list_gmsd=[]  
list_aver=[]  
list_his=[] 

for ind in range(12,18): 
    im1 = Image.open(r'E:\data\after_spm\OUTPUT_REAL\normalized\%d.jpg'%(ind))   #COMPARED IMAGES
    im2 = Image.open(r'E:\data\after_spm\OUTPUT_REAL\XI\%d.jpg'%(ind))       #REFERENCE IMAGES
    im1 = im1.resize((256,256),Image.ANTIALIAS)
    im2 = im2.resize((256,256),Image.ANTIALIAS)
    im1_c = im1.convert('L')
    im1_a = np.array(im1_c)
    im1_a = np.squeeze(im1_a)
#    list_1.append(im1_a)
#    img_a = np.array(list_1)  #2Dto3D
    im2_c = im2.convert('L')
    im2_a = np.array(im2_c)
    im2_a = np.squeeze(im2_a) 
#    list_2.append(im2_a)
#    im2_a = np.array(list_2)    #2Dto3D

    #calculate the evaluation criteria  
#    mse = msee(im1_a,im2_a)
    mse = np.sum((im1_a - im2_a)**2.0) / 255.0 / 255.0  #method2
    list_mse.append(mse)
    ssim = compute_ssim(im1_a, im2_a)
    list_ssim.append(ssim)
#    psnr = 10 * np.log10(255 * 255 / mse)
    psnr = psnrr(im1_a, im2_a,data_range=256)     #method2
    list_psnr.append(psnr)
#    gmsd = gmsdd(im1_a,im2_a)
#    list_gmsd.append(gmsd)
    average_disparity = np.average(np.divide(np.abs(np.subtract(np.array(im1_a,dtype=float),np.array(im2_a,dtype=float))),(np.array(im2_a,dtype=float))+0.0001))
    list_aver.append(average_disparity)
    
    #calculate histogram correlation
    CNT_standard=cv2.calcHist([im1_a],[0],None,[256],[0,256])
    CNT_ref=cv2.calcHist([im2_a],[0],None,[256],[0,256])
    hist_correlation = cv2.compareHist(CNT_standard,CNT_ref,cv2.HISTCMP_BHATTACHARYYA)
    list_his.append(hist_correlation)

print('mse:',np.average(np.array(list_mse)))
print('ssim:',np.average(np.array(list_ssim)))
print('psnr:',np.average(np.array(list_psnr)))
#print('gmsd:',np.average(np.array(list_gmsd)))   #The code provided in the GMSD paper is the MATLAB version, and the MATLAB version has been uploaded.
print('aver:',np.average(np.array(list_aver)))
print('hist:',np.average(np.array(list_his)))


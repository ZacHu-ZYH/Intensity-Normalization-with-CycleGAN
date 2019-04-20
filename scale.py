# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 21:26:47 2019

@author: Administrator
"""

import os
import tensorflow as tf
import cv2
from PIL import Image
 
#读取图像

for m1 in range(0,19):
    image = Image.open(r"F:\data\after_spm\OUTPUT_REAL\GE\%d.jpg"%m1)
    size = (256,256)
    image = image.resize((256, 256))
    image.save(r"F:\data\after_spm\OUTPUT_REAL\GE\new_%d.jpg"%m1)
    image = Image.open(r"F:\data\after_spm\OUTPUT_REAL\XI\%d.jpg"%m1)
    size = (256,256)
    image = image.resize((256, 256))
    image.save(r"F:\data\after_spm\OUTPUT_REAL\XI\new_%d.jpg"%m1)
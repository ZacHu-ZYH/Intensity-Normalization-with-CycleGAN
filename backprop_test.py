# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:07:59 2019

@author: Administrator
"""
import keras.backend as K
import tensorflow as tf
import numpy as np

value_range = [0.0, 5.0]
a = np.array([-1.0, 0.0, 1.5, 2.0, 5.0, 15])

x = K.variable(a)
y = 2*x
forward_loss = tf.reduce_mean(tf.abs(x-y))
backward_loss = tf.reduce_mean(tf.abs(x+y))
loss = forward_loss+backward_loss
hist = tf.histogram_fixed_width(x, value_range, nbins=5, dtype=tf.int32)
##########直方图函数能否反向传播验证###############
try:
    gradient = K.gradients(hist, x)
    print('gradient:',gradient)
except:
    pass
##########CycleGAN中循环一致性损失函数能否反向传播验证###############
gradient1 = K.gradients(loss, x)
print('gradient1',gradient1)
##########numpy分位数函数能否反向传播验证###############
percent = np.percentile(x,50)
mm = tf.bitcast(x,'int32')
tt = tf.bincount(mm)
print(tt)
gradient2 = K.gradients(percent, x)
print('gradient2',gradient2)

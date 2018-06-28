#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 16:40:40 2018

@author: hari
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import random

#%%
df_train = pd.read_csv("/home/hari/tensorflow/Capstone/labels.csv")

df_train = df_train[:]
df_train.head()

#%%
# Reshaping the images 
img_width=200
img_height=200
images=[]
classes=[]
#load training images
for f, breed in tqdm(df_train.values):
    img = cv2.imread('/home/hari/tensorflow/Capstone/train/{}.jpg'.format(f))
    classes.append(breed)
    images.append(cv2.resize(img, (img_width, img_height)))


#%%
# Changing color
#images_aug = cv2.bilateralFilter(images[1],9,75,75)
inputs = []
for i in range(10222):
     inputs.append(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
     
#%%
# Image blurring
random_numbers1 = [np.random.randint(0, len(images)) for p in range(0,4000)]

flip_inputlr = []
flip_classlr =[]
for i in random_numbers1:
    flip_inputlr.append(cv2.flip(inputs[i],1))
    flip_classlr.append(classes[i])
len(flip_inputlr)
flip_lr_arr = np.array(flip_inputlr)
#flip_class_lr_arr = np.array(flip_classlr)
#%%
# Flipping 
random_numbers2 = [np.random.randint(0, len(images)) for p in range(0,4000)]

flip_inputup = []
flip_classup = []
for i in random_numbers2:
    flip_inputup.append(cv2.flip(inputs[i],0))
    flip_classup.append(classes[i])
len(flip_inputup)

flip_up_arr = np.array(flip_inputup)
#flip_class_up_arr = np.array(flip_classup)

#%%
final = np.concatenate((flip_lr_arr, flip_up_arr), axis=0)

flip_classlr.extend(flip_classup)
#flip_classlr.extend(blur_class)

print(flip_classlr)

print(final.shape)
id = np.arange(8000)
print(len(id))
#%%

final1=[]
for i in range(final.shape[0]):
    final1.append(cv2.imwrite('/home/hari/tensorflow/Capstone/train/{}.jpg'.format(i),final[i]))

labels1 = pd.DataFrame({'id':id, 'breed':flip_classlr},columns=['id','breed'])

labels1.to_csv('/home/hari/tensorflow/Capstone/labels1.csv', index=False)

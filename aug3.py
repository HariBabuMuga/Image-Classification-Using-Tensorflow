#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 17:12:23 2018

@author: hari
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

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
random_numbers5 = [np.random.randint(0, len(images)) for p in range(0,4000)]

kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
kernel_crossed = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

Erosion_input =[]
Erosion_class=[]
for i in random_numbers5:
    Erosion_input.append(cv2.erode(inputs[i],kernel_rect,iterations=1))
    Erosion_class.append(classes[i])
erosion_arr=np.array(Erosion_input)
#blur_class_arr = np.array(blur_class)

#%%
random_numbers6 = [np.random.randint(0, len(images)) for p in range(0,4000)]

Dilate_input =[]
Dilate_class=[]
for i in random_numbers6:
    Dilate_input.append(cv2.dilate(inputs[i],kernel_ellipse,iterations=1))
    Dilate_class.append(classes[i])
Dilate_input_arr = np.array(Dilate_input)

#%%
final = np.concatenate((erosion_arr, Dilate_input_arr), axis=0)

Erosion_class.extend(Dilate_class)
#flip_classlr.extend(blur_class)

print(Erosion_class)

print(final.shape)
id = np.arange(16000, 24000)
print(id[2])

#%%
for i in range(final.shape[0]):
    print(i)
#%%

final1=[]
for i in range(final.shape[0]):
    j = i + 16000
    final1.append(cv2.imwrite('/home/hari/tensorflow/Capstone/train/{}.jpg'.format(j),final[i]))

labels3 = pd.DataFrame({'id':id, 'breed':Erosion_class}, columns=['id','breed'])

labels3.to_csv('/home/hari/tensorflow/Capstone/labels3.csv', index = False)


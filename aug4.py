#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 17:24:41 2018

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
random_numbers7 = [np.random.randint(0, len(images)) for p in range(0,4000)]

kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
kernel_crossed = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

Opening_input =[]
Opening_class=[]
for i in random_numbers7:
    Opening_input.append(cv2.morphologyEx(inputs[i],cv2.MORPH_OPEN,kernel_crossed))
    Opening_class.append(classes[i])

dilation_arr=np.array(Opening_input)
#blur_class_arr = np.array(blur_class)

#%%
random_numbers8 = [np.random.randint(0, len(images)) for p in range(0,4000)]

Closing_input = []
Closing_class=[]
for i in random_numbers8:
    Closing_input.append(cv2.morphologyEx(inputs[i],cv2.MORPH_CLOSE,kernel_ellipse))
    Closing_class.append(classes[i])

closing_arr=np.array(Closing_input)

#%%
final = np.concatenate((dilation_arr, closing_arr), axis=0)

Opening_class.extend(Closing_class)
#flip_classlr.extend(blur_class)

print(Opening_class)

print(final.shape)
id = np.arange(24000, 32000)
print(id[2])

#%%
for i in range(final.shape[0]):
    print(i)
#%%

final1=[]
for i in range(final.shape[0]):
    j = i + 24000
    final1.append(cv2.imwrite('/home/hari/tensorflow/Capstone/train/{}.jpg'.format(j),final[i]))

labels4 = pd.DataFrame({'id':id, 'breed':Opening_class}, columns=['id','breed'])

labels4.to_csv('/home/hari/tensorflow/Capstone/labels4.csv', index = False)


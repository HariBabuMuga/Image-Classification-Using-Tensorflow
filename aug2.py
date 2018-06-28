#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 17:03:28 2018

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
# Normal Blur
random_numbers3 = [np.random.randint(0, len(images)) for p in range(0,4000)]

blur_input = []
blur_class = []
for i in random_numbers3:
    blur_input.append(cv2.blur(inputs[i],(5,5)))
    blur_class.append(classes[i])

blur_arr = np.array(blur_input)
#blur_class_arr = np.array(blur_class)

#%%
random_numbers4 = [np.random.randint(0, len(images)) for p in range(0,4000)]

MedianBlur_input =[]
MedianBlur_class=[]
for i in random_numbers4:
    MedianBlur_input.append(cv2.medianBlur(inputs[i],5))
    MedianBlur_class.append(classes[i])

Median_Blur_arr=np.array(MedianBlur_input)

#%%
final = np.concatenate((blur_arr, Median_Blur_arr), axis=0)

blur_class.extend(MedianBlur_class)
#flip_classlr.extend(blur_class)

print(blur_class)

print(final.shape)
id = np.arange(8000, 16000)
print(id[2])

#%%
for i in range(final.shape[0]):
    print(i)
#%%

final1=[]
for i in range(final.shape[0]):
    j = i + 8000
    final1.append(cv2.imwrite('/home/hari/tensorflow/Capstone/train/{}.jpg'.format(j),final[i]))

labels2 = pd.DataFrame({'id':id, 'breed':blur_class}, columns=['id','breed'])

labels2.to_csv('/home/hari/tensorflow/Capstone/labels2.csv', index = False)


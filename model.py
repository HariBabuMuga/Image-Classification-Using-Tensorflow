
# coding: utf-8

# In[1]:

from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import pandas as pd
import imgaug as ia
import os, sys
from tqdm import tqdm
import cv2
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg


# In[2]:

# setting path to read images
train_path = '/home/hari/tensorflow/Capstone/train/'
test_path = '/home/hari/tensorflow/Capstone/test/'

labels = pd.read_csv("/home/hari/tensorflow/Capstone/labels.csv")
labels1 = pd.read_csv("/home/hari/tensorflow/Capstone/labels1.csv")
labels2 = pd.read_csv("/home/hari/tensorflow/Capstone/labels2.csv")
labels3 = pd.read_csv("/home/hari/tensorflow/Capstone/labels3.csv")
labels4 = pd.read_csv("/home/hari/tensorflow/Capstone/labels4.csv")

Labels = pd.concat([labels,labels1,labels2,labels3,labels4])

print ('The train data has {} images.'.format(Labels.shape[0]))
print(Labels.columns)

print("Total number of unique labels: " + str(Labels['breed'].nunique()))
print("Total count of each category") 
# Counting each breed of dogs
print(Labels["breed"].value_counts())

id = Labels['id']
print(id)
# In[3]:

print(Labels["breed"].value_counts())
Labels["breed"] = Labels["breed"].astype('category')

# Integer encoding
Labels["breed"] = Labels["breed"].cat.codes   

# onehot encoding 
y_onehot = pd.get_dummies(Labels['breed'])
print(y_onehot)


# In[4]:

# function to read imges into arrays
def read_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = img.astype(np.float32)
    img = cv2.resize(img, (200, 200))
    return(img)


train_labels = Labels['breed'].values
#print(" Names of all the breeds of dogs")
#print(train_labels)


# In[9]:
#We upload all the packages we need
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import math
import os
import scipy.misc
from scipy.stats import itemfreq
from random import sample
import pickle
from sklearn.metrics import confusion_matrix

# Image manipulation.
import PIL.Image
from IPython.display import display
#from resizeimage import resizeimage

#Panda
import pandas as pd
import time


# In[10]:

# Building neural network 
# Training Parameters# Train 
learning_rate = 0.001
#num_steps = 10
epochs = 15

batch_size = 100
display_step = 10

# Network Parameters
img_height = 200 # MNIST data input (img shape: 28*28)
img_width = 200
num_classes = 120 # MNIST total classes (0-9 digits)
dropout = 0.5 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, img_height,img_width, 3])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


# In[11]:

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='VALID')


# In[12]:

# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    #x = tf.reshape(x, shape=[-1, 160, 160, 3])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    
    # Convolution Layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=2)


    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])

    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    #fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# In[13]:

# Store layers weight & bias# Store 
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.1)),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([3, 3, 32, 64],stddev=0.1)),
        # 5x5 conv, 32 inputs, 64 outputs
    'wc3': tf.Variable(tf.random_normal([3, 3, 64, 64],stddev=0.1)),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([25*25*64, 1024],stddev=0.1)),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes],stddev=0.1))
}

biases = {
    'bc1': tf.Variable(tf.constant(0.0, shape=[32])),
    'bc2': tf.Variable(tf.constant(0.0, shape=[64])),
    'bc3': tf.Variable(tf.constant(0.0, shape=[64])),
    'bd1': tf.Variable(tf.constant(0.0, shape=[1024])),
    'out': tf.Variable(tf.constant(0.0, shape=[num_classes]))
}

# In[14]:

# Construct model
logits = conv_net(X, weights, biases, keep_prob)
#prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 
tf.summary.scalar('loss', loss_op)
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

#%%

# Add ops to save and restore all the variables.
saver = tf.train.Saver(max_to_keep=2)
#writer = tf.summary.FileWriter('/home/hari/tensorflow/Capstone', tf.get_default_graph())

#%%

def input(image_path,s,e):
    # converting image into an array of pixels
    train_data=[]   
    for img in Labels['id'][s:e]:
        train_data.append(read_image(train_path + '{}.jpg'.format(img)))
        #plt.imshow(read_image(train_path + '{}.jpg'.format(img)))
        #print(np.array(train_data).shape)
    train_data = np.array(train_data)/255
    #print(train_data.shape)
    #plt.imshow(train_data[4,:,:,:])
        #s = e 
        #e += 500
    #x = train_data.reshape(train_data.shape[0], train_data.shape[1]*train_data.shape[2],train_data.shape[3])
    #print(train_data.shape)
    #plt.imshow(x[4,:,:])
    y = y_onehot[s:e]
    #print(y.shape)
    return(train_data,y)
 
    
#%%
    
"""
kl,l = input(train_path,10,35)
#print(kl)
#for i in kl:
#    plt.imshow(i)

plt.imshow(kl[19]/10)
#plt.show()
print(kl[3])

"""
#%%

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    total_loss = []
    prev_accu = 0

    for epoch in range(epochs):
        s = 0
        e = batch_size
        losses = 0

        for i in range(int(len(Labels['id'])/batch_size)):
            print()
            print("Batch From  " + str(s) + "  To  " + str(e))
            print()
            
            batch_x, batch_y = input(train_path,s,e)
            
            
            _, loss, acc = sess.run([train_op,loss_op,accuracy], feed_dict={X: batch_x, Y: batch_y})  # keep_prob: dropout
            s = e
            e += batch_size
            
            print("Epoch- " + str(epoch) + ", Batch- " + str(i) + ", Minibatch Loss = " +                   "{:.4f}".format(loss) + ", Training Accuracy = " +                   "{:.3f}".format(acc))

            
            losses += loss
            test_accuracy = 0
            num = 0
            if e > 35000 and e<= 35105:
                
                print("Optimization Finished!")
                
                k = s
                for j in range(int(len(Labels['id'][k:])/batch_size)):
                    
                    print()
                    print("Batch From  " + str(s) + "  To  " + str(e))
                    print()
                
                    batch_X, batch_Y = input(train_path,s,e)
                    test_ac = sess.run(accuracy, feed_dict={X: batch_X, Y: batch_Y})
                    test_accuracy += test_ac
                    num += 1
                    
                    s = e
                    e += batch_size
                    test_accu = (test_accuracy/num)
                    
                    print("Test Accuracy of Batch " + str(j) + " is  " + str(test_accu))
                    
                    if e >= 42200 and e <= 42222:
                        break
                    
                print("-"*80)
                print("Test Accuracy after epoch " + str(epoch) + " is  " + str(test_accu)) #keep_prob: .75
                print("-"*80)

                    
            if e >= 42200 and e <= 42222:
                break
                
        print("Epoch Number" + str(epoch) + ", Minibatch Loss= " +   "{:.4f}".format(loss) + ", Training Accuracy= " +  "{:.3f}".format(acc))    
        
        total_loss.append(losses)
        
        """
        # Calculate accuracy for 256 MNIST test images
        
        
        
        batch_X, batch_Y = input(train_path,7101,10222)
        
        
        test_accu = sess.run(accuracy, feed_dict={X: batch_X,
                                      Y: batch_Y})
    
        """
        cur_accu = test_accu
        
        if cur_accu > prev_accu:
            
            save_path = saver.save(sess, "/home/hari/tensorflow/Capstone/test_model", global_step = epoch, write_meta_graph=True)
            
        prev_accu = cur_accu
    

#writer.close()
#%%
        
#import os 
#os.system('tensorboard --logdir=/home/hari/tensorflow/Capstone --port 6006')

#%%
plt.figure()
plt.plot(total_loss, np.arange(epochs), label = 'Loss Curve')
plt.legend()
plt.show()

#%%
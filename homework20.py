#!/usr/bin/env python
# coding: utf-8

# **Homework 20**

# In the next assignment you will create a Convolutional Neural Network to do facial recognition. This will use a large dataset of face photos, which we'll explore here. As this is a very large dataset, and may take a while to load. Once it is complete we will just look at a subset, consisting of people for whom there are at least 70 photos.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


from sklearn.datasets import fetch_lfw_people
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
faces=lfw_people.images
names=lfw_people.target_names
target=lfw_people.target

for name in names:
  print(name)

faces.shape


# You see there are a total of 1288 images, each is 50-by-37 pixels, representing seven people. Let's take a look at one:

# In[2]:


plt.imshow(faces[10],cmap='gray')


# In[3]:


target[10]


# You see the target for image 10 is the number 3. Notice that President George W. Bush (the man in the photo) is the 3rd name on the list above (counting from 0). For convenience, we will name this image `bush`:

# In[4]:


bush=faces[10]


# To create a CNN, you must understand two operations: convolution and pooling. Write a `Conv` function that takes two arguments, image and filter. Both will be 2-dimensional numpy arrays. Your function should return the convolution of the image array by the filter array.

# In[5]:


def Conv(image,kernel):
    i_height, i_width = image.shape
    k_height, k_width = kernel.shape

    output_height = i_height - k_height + 1
    output_width = i_width - k_width + 1
    output = np.zeros((output_height, output_width))
    for i in range(output_height):
        for j in range(output_width):
            section = image[i:i+k_height, j:j+k_width]
            output[i, j] = np.sum(section * kernel)

    return output


# To see the effect of your code, we define a 7-by-7 kernel:

# In[6]:


kernel=np.zeros((7,7))
kernel[3,:]=1
kernel


# We now apply this filter to the image of George Bush.

# In[7]:


plt.imshow(Conv(bush,kernel),cmap='gray')


# You can see this kernel has the effect of horizontally smearing the image.

# The next element of a CNN is a way to downsample the image to something of lower resolution. Implement a `MaxPool` function which takes an image and a tuple called "pool_size". If the pool_size is (n,m), then the function should output a lower resolution image where each n-by-m window of the original is replaced by a single pixel whose intensity is the maximum value in the window.

# In[8]:


def MaxPool(image,pool_size):
    pool_height, pool_width = pool_size
    i_height, i_width = image.shape

    output_height = i_height // pool_height
    output_width = i_width // pool_width

    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            section = image[i*pool_height:(i+1)*pool_height, j*pool_width:(j+1)*pool_width]
            output[i, j] = np.max(section)
    return output


# We can see the effect of this by applying it to the smeared image of Bush:

# In[9]:


plt.imshow(MaxPool(Conv(bush,kernel),(2,2)),cmap='gray')


# Most of the features are now gone, but the basic mouth shape is still there. Hence, this particular kernel, followed by a MaxPooling, may be good at picking out mouth shapes. A different kernel might be useful for picking out eye shapes, nose shapes, etc.

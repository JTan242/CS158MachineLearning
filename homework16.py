#!/usr/bin/env python
# coding: utf-8

# **Homework 16**

# For this assignment, all you'll need is numpy:

# In[1]:


import numpy as np


# In this assignment, you'll be creating three classes: `Linear`, `ReLU`, and `Sequential`. The `Linear` and `ReLU` classes will define layers of a Neural Network. The `Sequential` class defines a particular network, given a list of layers.
# 
# For example, consider the following code:
# ```
# layer1=Linear(2,3)
# layer2=ReLU()
# layer3=Linear(3,1)
# network=Sequential([layer1,layer2,layer3])
# ```
# 
# Here, layer1 takes a feature matrix with 2 columns (features), and produces a matrix with the same number of rows and 3 columns by a simple linear function defined by 2 coefficients (the "weights" or "kernel") and 1 intercept (the "bias") . You should visualize this as two neurons feeding into 3 neurons. layer2 accepts the output of layer 1, and produces an output of similar size where all of the negative entries have been set to 0. This is considered an "activation" layer. Incluion of such a layer is what makes the network capable of modeling non-linear data. Finally, the last layer produces a single output for each observation.
# 
# All three classes (Sequential, Linear and ReLU) should have a `__call__` method. If `X` is a feature matrix of shape (n,2), then running `network(X)` after the above code will call the `__call__` methods of each layer, and produce a matrix of shape (n,1).
# 

# In[2]:


class Linear():
    '''Fully connected linear layer class'''
    def __init__(self, input_size, output_size):
        np.random.seed(1) #Don't use in practice! This is just to make sure we all get the same answers
        self.kernel = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size) #Standard initialization of weights
        self.bias = np.zeros(output_size) #Standard initialization of intercept

    def __call__(self,input):
        return np.dot(input, self.kernel) + self.bias


# In[3]:


class ReLU():
    '''ReLU layer class'''
    def __init__(self):
        pass #No init function necessary

    def __call__(self,input):
        return np.maximum(0, input)


# In[4]:


class Sequential():
    def __init__(self,layerlist):
        self.layerlist=layerlist

    def __call__(self,input):
        for layer in self.layerlist:
            input = layer(input)
        return input


# Now test your code. Run this code block. You should see a matrix of shape (15,1).

# In[8]:


np.random.seed(4)
X=np.random.random(30).reshape((15,2)) #generate a random feature matrix with 2 features, and 15 observations

layer1=Linear(2,3)
layer2=ReLU()
layer3=Linear(3,1)
network=Sequential([layer1,layer2,layer3])

network(X)


# In[ ]:





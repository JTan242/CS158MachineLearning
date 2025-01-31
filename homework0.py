#!/usr/bin/env python
# coding: utf-8

# CSCI 158 **HOMEWORK 0**
# 
# In this homework assignment we explore the Numpy python package. For a brief introduction, go [here](https://numpy.org/devdocs/user/absolute_beginners.html).
# 
# Begin by importing the Numpy library:

# In[18]:


import numpy as np


# Create a numpy array called `X` with 100 rows and 10 columns, where each entry is the number 1.

# In[19]:


X = np.ones((100, 10))
X


# Create a 1-dimensional numpy array called `y`, where the entries are the numbers 1 through 1000 (in order).

# In[20]:


y = np.arange(1, 1001)
y


# Create a numpy array called `Y` with 100 rows and 10 columns, whose entries come from the array `y`.

# In[21]:


Y=y[:1000].reshape(100, 10)
Y


# Use the np.newaxis command to create an array called `z`, with enties from `y`, which has 1000 rows and one column.

# In[22]:


z=y[:, np.newaxis]
z


# Create an array of shape (5,4) called `A`, consisting of those entries in `Y` in the first 5 rows and columns 2-5.  

# In[24]:


A=Y[:5, 2:6]
A


# Create a 1-dimensional array called `b` consisting of the entries in `Y` that are divisible by 3, and between 20 and 70.
# 

# In[11]:


b=Y[(Y % 3 == 0) & (Y >= 20) & (Y <= 70)]
b


# Create an array called `Z` whose entries are the square-root of the entries in `Y`, plus the corresponding entries in `X`.

# In[13]:


Z=np.sqrt(Y) + X
Z


# Create a 1-D array `m` whose entries are the maximum of each column of `Y`.

# In[ ]:


m=np.max(Y, axis=0)
m


# Create a 1-D array `s` whose entries are the sums of each row of `Y`.

# In[ ]:


s=np.sum(Y, axis=1)
s


# Create an array `P` which is the product of `X` and the transpose of `Y`.

# In[17]:


P=np.dot(X, Y.T)
P


# In[ ]:





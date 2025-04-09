#!/usr/bin/env python
# coding: utf-8

# **Homework 19.**

# In this assignment you will train a Neural Network to recognize handwritten numbers. Begin by importing the older libraries we've used:

# In[8]:


import numpy as np
from matplotlib.pyplot import imshow
from sklearn.model_selection import train_test_split


# Now import the dataset:

# In[9]:


#Import data and target:
from sklearn.datasets import load_digits
digits=load_digits()
X=digits.data
y=digits.target

X.shape,y.shape


# You should see that there are 1797 observation in X, and each has 64 features. The features are the pixel values in an 8x8 image. For example, the pixel values for image 100 are given by:

# In[10]:


X[100,:]


# We can see what this data represents by reshaping it into an array of shape (8,8) and visualizing with the imshow command from matplotlib.pyplot:

# In[11]:


imshow(np.array(X[100,:]).reshape((8,8)),cmap='gray')


# This is a (very pixelized) image of the numeral 4. We can confirm this by looking at the 100th value of the target vector y:

# In[12]:


y[100]


# Our goal is to create and train a neural network that takes in pixel data and predicts the variable y, telling us what numeral the image represents. To this end, it will be helpful to normalize pixel values to the range (-1,1):

# In[13]:


X=(X-8)/8 #normalization--each pixel was in range 0-16


# We now do a train/test split. As usual, we'll train our model on the train set, and evaluate its performance on the test set.

# In[14]:


#Train/test split
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,train_size=0.8)


# Next we import the necessary libraries from pytorch, and convert all data to pytorch tensors:

# In[15]:


import torch
from torch import nn
from torch.optim import Adam

Xtrain=torch.tensor(Xtrain,dtype=torch.float32)
Xtest=torch.tensor(Xtest,dtype=torch.float32)
ytrain=torch.tensor(ytrain)
ytest=torch.tensor(ytest)


# Now, build a neural network. Your network should take in observations with 64 feaures, and generate an output of 10 numbers (one for each possible numeral that might be represented by the image). Use two hidden layers with 32 and 16 neurons, respectively. After each Linear layer, add a ReLU layer, Batch Normalization, and 10% dropout.

# In[21]:


#Build the network:
digitsNN=nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.BatchNorm1d(32),
    nn.Dropout(0.1),

    nn.Linear(32, 16),
    nn.ReLU(),
    nn.BatchNorm1d(16),
    nn.Dropout(0.1),

    nn.Linear(16, 10)
)


# Instantiate an Adam optimizer to use with your model, with a learning rate of 0.01.

# In[22]:


optimizer=Adam(digitsNN.parameters(), lr=0.01)


# Write a training loop to train your model, using 1000 epochs of batch gradient descent, with batches of size 100.

# In[23]:


n_epochs=1000
N = Xtrain.shape[0]  # total number of observations in training data
batch_size=100
loss_fn = nn.CrossEntropyLoss()
for epoch in range(n_epochs):
  # Shuffle the indices
  indices = torch.randperm(N)

  # Create mini-batches
  for i in range(0, N, batch_size):
    batch_indices = indices[i:i+batch_size]
    batch_X = Xtrain[batch_indices]
    batch_y = ytrain[batch_indices]

    #YOUR CODE HERE
    y_pred = digitsNN(batch_X)
    CEloss = loss_fn(y_pred, batch_y)
    optimizer.zero_grad()
    CEloss.backward()
    optimizer.step()

  if epoch%100==0:
    print(f"epoch: {epoch}, loss: {CEloss.item()}")


# Next, report the accuracy of your model on the test set. Note the use of `with torch.no_grad()` here, because there is no longer a need to track gradients once our model is trained.

# In[26]:


with torch.no_grad():
  y_pred=digitsNN(Xtest) #generate predictions for the test set.
    
predicted_labels = torch.argmax(y_pred, dim=1)
accuracy = (predicted_labels == ytest).float().mean().item()
# accuracy=#YOUR CODE HERE
accuracy


# Copy and paste this into homework19gradescope.ipynb, export that as a python file, and upload to gradescope.

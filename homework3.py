#!/usr/bin/env python
# coding: utf-8

# **Homework 3**
# 
# Load the iris dataset:

# In[15]:


import numpy as np


# Create a PCA class that is instantiated by specifying the number of desired components.

# In[12]:


class PCA():
  def __init__(self,n_components):
    self.n_components=n_components

  def fit(self,X):
    self.mean = np.mean(X, axis=0)
    centeredX = X - self.mean
    CVmatrix= np.cov(centeredX, rowvar=False)
    eigvals,eigvecs= np.linalg.eig(CVmatrix)

    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]  
    #Create a basis from the eigenvecs with the largest corresponding eigenvals:
    self.basis = eigvecs[:, :self.n_components]
      
  def transform(self,X):
    centeredX = X - self.mean
    return centeredX @ self.basis

  def fit_transform(self,X):
    self.fit(X)
    return self.transform(X) #Combines the fit method and the transform method for convenience


# The following code block loads the  `iris` dataset and applies a `PCA` object with 2 components.

# In[13]:


from sklearn.datasets import load_iris

iris=load_iris()
X=iris.data
y=iris.target

pca=PCA(n_components=2)
projectedX=pca.fit_transform(X)


# Run this code block to visualize your projection! Note that the color of each point comes from the species, allowing you to see to what extent those points form distinct clusters.

# In[14]:


import matplotlib.pyplot as plt

plt.scatter(projectedX[:,0],projectedX[:,1],c=y)


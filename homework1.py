#!/usr/bin/env python
# coding: utf-8

# **Homework 1**

# We begin with the usual import, and a new one:

# In[ ]:


import numpy as np
from sklearn.datasets import load_iris


# Now load the iris dataset.

# In[ ]:


iris=load_iris()
X=iris.data 
y=iris.target


# The columns of the numpy array `X` (our "feature matrix") give the Sepal Length, Sepal Width, Petal Length and Petal Width of 150 different observed iris flowers. `y` is our "target", an array of 150 integers indicating the specific species of iris, where 0=Setosa, 1=Versicolor, and 2=Virginica.
# 
# Here are the first few rows of `X`:

# In[ ]:


X[:5,:]


# For this assignment, we'll only work with the Petal Length and Petal Width of each flower, so we can redefine `X` to be just the last two columns:

# In[ ]:


X=X[:,2:]
X.shape


# Define a function `sq_distances` with inputs `X` (a numpy array with two columns), `length` and `width` (the Petal Length and Petal Width of an unknown flower). The function should return an array of squared distances from the unknown point to each point in `X`. Use vectorized Numpy operations, NOT A FOR LOOP. 

# In[ ]:


def sq_distances(X,length,width):
    return (X[:, 0] - length)**2 + (X[:, 1] - width)**2


# Define a function `SpeciesOfKNeighbors` that gives the species label (a number 0, 1, or 2) of the k nearest neighbors from the point with given Petal Length and Petal Width to the points in `X`. (The list of species labels for each point in `X` is contained in the array `y`.) *Hint: The numpy function `argsort()` is useful for this problem.*

# In[ ]:


def SpeciesOfNeighbors(X,y,length,width,k):
    distances = sq_distances(X, length, width)
    nearest_indices = distances.argsort()[:k]
    return y[nearest_indices]


# Create a function `majority` that takes an array of labels, and returns the label that appears the most often. *Hint: The numpy functions `bincount()` and `argmax()` can be useful here.*

# In[ ]:


def majority(labels):
    label_counts = np.bincount(labels)
    most_common_label = np.argmax(label_counts)
    return most_common_label


# Combine your previous functions to create a function `KNN` which takes a feature matrix `X` of known Petal Lengths and Petal Widths, a target array `y` containing their species labels, a hyperparameter `k`, and the `length` and `width` of the petal of an unknown flower. Your function should return the most common species index among the k nearest neighbors of the unknown flower. 

# In[ ]:


def KNN(X,y,length,width,k):
  nearest_neighbors_labels = SpeciesOfNeighbors(X,y,length,width,k)
  most_common_label = majority(nearest_neighbors_labels)
  return most_common_label


# Test your code by playing with a few values for length, width, and k. For example, try:

# In[ ]:


KNN(X,y,1,1,7)


# Moving forward, we'll write our ML models as classes that conform to the standards of the sklearn package. Let's do this now. Modify your functions above to create appropriate methods for the following class:

# In[ ]:


class KNeighborsClassifier():
    def __init__(self,k):
        self.n_neighbors=k

    def fit(self,X,y):
        self.X=X
        self.y=y

    def sq_distances(self,length,width):
      return (self.X[:, 0] - length)**2 + (self.X[:, 1] - width)**2


    def SpeciesOfNeighbors(self,length,width):
      distances = self.sq_distances(length, width)
      nearest_indices = distances.argsort()[:self.n_neighbors]
      return self.y[nearest_indices]

    def majority(self,labels):
      label_counts = np.bincount(labels)
      most_common_label = np.argmax(label_counts)
      return most_common_label

    def predict(self,length, width):
      nearest_neighbors_labels = self.SpeciesOfNeighbors(length, width)
      return self.majority(nearest_neighbors_labels)


# If done correctly, the following code should produce the same answer as before:

# In[ ]:


knn=KNeighborsClassifier(7)
knn.fit(X,y)
knn.predict(1,1)


#!/usr/bin/env python
# coding: utf-8

# **Homework 2**

# This assignment is a continuation of homework 1. We begin with the usual imports.

# In[4]:


import numpy as np
from sklearn.datasets import load_iris


# We now load the iris dataset and create both feature matrix and target array. This time, we will use all four features, rather than just Petal Length and Petal Width.

# In[5]:


iris=load_iris()
X=iris.data
y=iris.target


# In the previous assignment you created a KNeighborsClassifier class. Here we'll load a pre-written version of this class from scikit-klearn. The syntax for its usage is exactly the same as for the one you wrote. I encourage you to read the full API for this implementation [here](https://scikit-learn.org/1.6/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).

# In[6]:


from sklearn.neighbors import KNeighborsClassifier


# The iris dataset contains 150 observations. We'd like to set aside 20% of these for testing the accuracy of our model(s). To this end, we'll create a Numpy array `test_indices` with a random sample of 20% of the numbers from 0 to 149. Then, we create a boolean Numpy array `test_mask` with a value of True for each index listed in `test_indices`, and False for the other values. Finally, we create a boolean Numpy array `train_mask` with the negation of each entry in `test_mask`. MAKE SURE YOU UNDERSTAND EACH LINE OF CODE.

# In[8]:


np.random.seed(6) #controls randomness (Don't change this!)
size=len(X)  #size of original dataset (should be 150 for iris)
test_frac=0.2 #fraction of dataset to set aside for testing
test_size=int(size*test_frac) #desired size of test dataset
test_indices=np.random.choice(np.arange(size),test_size) #random sample of indices from iris
test_mask=np.zeros(size,dtype=bool) #numpy array of False values
test_mask[test_indices]=True #change values as desired indices to True
train_mask=~test_mask #True->False, False->True


# Define `test_data` to be a feature matrix consisting of only those rows of `X` specified by `test_mask`. Define `test_target` to be an array containing the corresponding entires in `y`. Define `train_data` and `train_target` similarly.

# In[9]:


train_data=X[train_mask]
train_target=y[train_mask]
test_data=X[test_mask]
test_target=y[test_mask]


# Define a function called `predict_labels` whose inputs are `train_data`, `train_target`, `test_data` and `k`. Your function should create an instance of the KNeighborsClassifier class with the appropriate value of k, and use it to output an array of predicted labels (one for each entry in `test_data`) based on the k-closest points in train_data.

# In[10]:


def predict_labels(train_data,train_target,test_data,k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, train_target)
    return knn.predict(test_data)


# Define a function called `accuracy` whose inputs are `train_data`, `train_target`, `test_data`, `test_target` and `k`. Your function should return the accuracy: the fraction of times your `predict_labels` function returned the correct answer.

# In[20]:


def accuracy(train_data,train_target,test_data,test_target,k):
    predictions=predict_labels(train_data,train_target,test_data,k)
    num_correct = sum(predictions == test_target)
    return num_correct / len(test_target)


# Our goal is to visualize the accuracy of the KNN algorithm for various values of k, so we may pick the best one. Reasonable values of k start at 1, and may go as high as 20 (depending on the application). For each such value of k, compute the accuracy and assemble these in a 1D Numpy array.

# In[21]:


k=np.arange(1,20) #possible values for k
accuracies = np.array([accuracy(train_data, train_target, test_data, test_target, k) for k in k])


# Run the following code block to visualize:

# In[22]:


import matplotlib.pyplot as plt
plt.plot(k,accuracies)


# The optimal value of k will be the first value that gives a maximum (think about why). What is it?

# In[ ]:


k=5


# What is the accuracy for this value of k?

# In[27]:


best_accuracy= accuracy(train_data, train_target, test_data, test_target, 5)
print(best_accuracy)


# In[ ]:





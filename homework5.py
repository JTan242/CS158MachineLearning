#!/usr/bin/env python
# coding: utf-8

# **Homework 5**

# In[158]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# We'll continue working with the wine dataset:

# In[159]:


wine=load_wine()
X=wine.data
y=wine.target


# As usual, we'll start with a train-test split:

# In[160]:


train_data, test_data, train_target, test_target = train_test_split(
    X, y, test_size=0.2, random_state=6)


# In this assignment, you'll build a Random Forest classifier, using Decision Trees. Rather than importing your DecisionTreeClassifier class from the previous assignment, I've imported such a class from sklearn in the first code cell above. It works ALMOST identically. There are two differences:
# 1. To instantiate the class for trees with a max_depth of 5, you have to explicitly say:
# 
# `tree=DecisionTreeClassifier(max_depth=5)`
# 
# rather than just:
# 
# `tree=DecisionTreeClassifier(5)`
# 
# 2. To make predictions for each row of a dataset, you can call
# 
# `tree.predict(X)`
# 
# However, to make predictions for a single data point x (e.g. x=X[0,:]), you'll have to promote it to a 2D array. One way to do this is
# 
# `tree.predict(x[np.newaxis,:)`
# 
# 
# 
# 

# Complete the code below to create a Random Forest Classifier. As before, we wrap this in a python class where the `__init__` function just records hyperparameters (in this case `max_depth` and `n_estimators`), the `fit` method is what creates the model from training data, and the `predict` method is what generates a predicition for an unknown data point.

# In[161]:


class RandomForestClassifier():
    def __init__(self,max_depth,n_estimators):
        self.max_depth=max_depth
        self.n_estimators=n_estimators

    def fit(self,X,y):
        self.trees=[] #A list of (tree,features) tuples
        for i in range(self.n_estimators):
            rows,cols=X.shape
            np.random.seed(i) #only for autograding purposes!!
            samples=np.random.choice(range(rows),rows,replace=True)
            #`Samples` are randomly selected row numbers (with replacement)
            features=np.random.choice(range(cols),int(np.sqrt(cols)),replace=False)
            #`features` are randomly selected column numbers
            tree=DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X[samples][:, features], y[samples])  
            self.trees.append((tree,features))

    def predict(self,x):
        preds=[] #preds will contain a prediction for each tree in forest
        for i in range(self.n_estimators):
            tree, features = self.trees[i]  # Retrieves the tree and training features
            x_sub = x[features]
            preds.append(tree.predict([x_sub])[0])
        return max(set(preds), key=preds.count)


# Now you are ready to create a Random Forest, based on the training data. Instantiate a Forest of 100 trees, each with a maximum depth of 5.

# In[162]:


forest = RandomForestClassifier(max_depth=5, n_estimators=100)


# Fit it to the training data and training target:

# In[163]:


forest.fit(train_data, train_target)


# Let's test your classifier on one point in the test set:

# In[164]:


forest.predict(test_data[30])


# Compare your answer to the actual label. Did your classifier get it right?

# In[165]:


test_target[30]


# Now we'll check the accuracy:

# In[166]:


accuracy=(test_target==np.apply_along_axis(lambda x:forest.predict(x),1,test_data)).sum()/len(test_target)
accuracy


# How does that compare to the accuracy of a single Decision Tree, found in the prevous assignment?

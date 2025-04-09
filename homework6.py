#!/usr/bin/env python
# coding: utf-8

# homework 6

# We begin by importing the regression versions of the models you learned about in previous assignments:

# In[1]:


import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


# In this assignment, we will work with the California housing dataset. Read about it [here](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset).

# In[2]:


from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
X, y = data.data, data.target
X.shape, y.shape


# As usual, we do a train/test split with 20% of our data set aside for the test set.

# In[3]:


train_data, test_data, train_target, test_target = train_test_split(X, y, test_size=0.20, random_state=42)


# Define a function `mse` that returns the mean squared error between known values of the target (`y_true`) and predicted values (`y_pred`):

# In[4]:


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Fit a Decision Tree Regressor to the training data. Use `random_state=42` and `max_depth=5`, so we all get the same answer. Then, use it to generate predictions for the test data and use your `mse` function to compute the mean squared error of those predictions.

# In[5]:


tree = DecisionTreeRegressor(max_depth=5, random_state=42)
tree.fit(train_data, train_target)
tree_mse= mse(test_target, tree.predict(test_data))
tree_mse


# Next, fit a Random Forest Regressor to the training data (use  `random_state=42`, `max_depth=5`, and `n_estimators=20`) and compute its mean squared error on the test set.

# In[6]:


forest = RandomForestRegressor(random_state=42, max_depth=5, n_estimators=20)
forest.fit(train_data, train_target)
forest_predictions = forest.predict(test_data)

forest_mse = mse(test_target, forest_predictions)
forest_mse


# The main purpose of this assignment is to code up a Gradient Boosted Tree regression model class. Such a model trains a sequence of decision trees, where each one predicts the error (residuals) of the sum of all previous trees (with a learning rate applied to each for stability). Setting `n_estimators=1` would  produce a single decision tree, identical to the one produced by the `DecisionTreeRegressor` class.
# 
# **Note:** Use `random_state=42` every time you initialize a Decision Tree.

# In[27]:


class GradientBoostingRegressor():
    def __init__(self, learning_rate, n_estimators, max_depth):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def fit(self, X, y):
        self.trees = [] 
        tree1 = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
        tree1.fit(X, y)
        self.trees.append(tree1)
        predictions = tree1.predict(X)
        residuals = y - predictions  

        for i in range(1, self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
            tree.fit(X, residuals)
            self.trees.append(tree)
            predictions += self.learning_rate * tree.predict(X)  
            residuals = y - predictions  

    def predict(self, X):
        predictions = self.trees[0].predict(X)
        for tree in self.trees[1:]: 
            predictions += self.learning_rate * tree.predict(X)
        return predictions


# Finally, fit a Gradient Boosted Tree regressor to the training data (use `learning_rate=0.5`, `n_estimators=20`, and `max_depth=5`) and compute its mean squared error on the test set.

# In[28]:


GBT = GradientBoostingRegressor(learning_rate=0.5, n_estimators=20, max_depth=5)
GBT.fit(train_data, train_target)
GBT_predictions = GBT.predict(test_data)
GBT_mse = mse(test_target, GBT_predictions)
GBT_mse


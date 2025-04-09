#!/usr/bin/env python
# coding: utf-8

# homework 15

# We begin by importing the regression versions of the models you learned about in previous assignments:

# In[10]:


import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


# In this assignment you'll code up a Gradient Boosting Classifier. Such a model trains a sequence of decision trees, where each one predicts the gradient of the (negative) log loss of $\sigma(t)$, where $t$ is the sum of all previous trees (with a learning rate applied to each for stability). Setting `n_estimators=1` would  produce a single decision tree classifier.

# For reference, here is code for a Gradient Boosting Regressor. You'll modify this to make the classifier.

# In[11]:


class GradientBoostingRegressor():
  def __init__(self,learning_rate, n_estimators, max_depth):
    self.learning_rate = learning_rate
    self.n_estimators = n_estimators
    self.max_depth = max_depth

  def fit(self, X, y):
    self.estimators = []
    tree = DecisionTreeRegressor(max_depth=self.max_depth,random_state=42)
    tree.fit(X, y)
    self.estimators.append(tree)
    current_prediction = tree.predict(X)

    for _ in range(1,self.n_estimators):
      residuals = y - current_prediction
      tree = DecisionTreeRegressor(max_depth=self.max_depth,random_state=42)
      tree.fit(X, residuals)
      self.estimators.append(tree)
      current_prediction += self.learning_rate * tree.predict(X)

  def predict(self, X):
    predictions=self.estimators[0].predict(X)
    for i in range(1,self.n_estimators):
      predictions += self.learning_rate * self.estimators[i].predict(X)
    return predictions


# To make this into a classifier, you'll need to define two functions:

# In[12]:


def sigmoid(t):
  '''returns the sigmoid of t'''
  return 1 / (1 + np.exp(-t))

def log_odds(p):
  '''returns the log odds of p'''
  return np.log(p / (1 - p))


# Now, modify the above regressor to make a classifier:

# In[17]:


class GradientBoostingClassifier():
  def __init__(self,learning_rate, n_estimators, max_depth):
    self.learning_rate = learning_rate
    self.n_estimators = n_estimators
    self.max_depth = max_depth

  def fit(self, X, y):
    self.estimators = []
    self.initial_prediction = log_odds(np.mean(y))
    current_prediction = np.full(len(y), self.initial_prediction)

    for _ in range(0,self.n_estimators):
      residuals = y - sigmoid(current_prediction)
      tree = DecisionTreeRegressor(max_depth=self.max_depth,random_state=1)
      tree.fit(X, residuals)
      self.estimators.append(tree)
      current_prediction += self.learning_rate * tree.predict(X)

  def predict_log_proba(self, X):
    '''returns the log of the probability of class 1, i.e., the
    output of the weighted sum of all trees'''
    predictions = self.initial_prediction*np.ones(len(X))
    for i in range(0,self.n_estimators):
      predictions += self.learning_rate * self.estimators[i].predict(X)
    return predictions

  def predict_proba(self, X):
    '''returns the probability of class 1, which is
    the sigmoid of the log probabilities'''
    return sigmoid(self.predict_log_proba(X))

  def predict(self, X):
    '''returns the class predictions:
    0 if prob<0.5, and 1 if prob>=0.5 '''
    return (self.predict_proba(X) >= 0.5).astype(int)



# To test the accuracy of your model, we'll create a challenging synthetic dataset:

# In[18]:


#Code courtesy of ChatGPT

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=10000,    # number of samples
    n_features=20,      # total number of features
    n_informative=10,   # number of features actually informative
    n_redundant=5,      # number of redundant features
    n_classes=2,        # binary classification
    random_state=42
)

train_data, test_data, train_target, test_target = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# Check the accuracy *on the test data* of a Gradient Boosting Classifier model with learning rate=0.1, 20 estimators, and a max_depth of 5 which has been trained *on the train data.*  

# In[19]:


# #YOUR CODE HERE
# accuracy=#YOUR CODE HERE
# accuracy
model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=20, max_depth=5)
model.fit(train_data, train_target)
predictions = model.predict(test_data)
accuracy = np.mean(predictions == test_target)
accuracy


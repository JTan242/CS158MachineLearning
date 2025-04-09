#!/usr/bin/env python
# coding: utf-8

# **Homework 7**

# In[79]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In this assignment we'll look at the classic "cars" dataset. You can read about it here:
# [link](https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/auto.html)

# In[80]:


cars=pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/auto.csv')
cars.head(3)


# At first, we'll just focus on finding a relationship between the `displacement` and `mpg` columns. 

# In[81]:


disp=np.array(cars.displacement)
mpg=np.array(cars.mpg)


# Let's take a look at them:

# In[82]:


plt.scatter(disp, mpg)
plt.xlabel('Displacement')
plt.ylabel('MPG')


# Problem 1.
# 
# Create a class that finds a linear model describing the relationship between arrays `X` and `y` (you may assume each are 1-dimensional). This model should be created using only the highest and lowest values in `X`, and the corresponding values in `y`.   

# In[83]:


class MaxMinLinearRegression():
    def __init__(self):
        '''No init function needed,
        since there are no hyperparameters'''
        pass

    def fit(self, X, y): 
        '''Stores the slope and intercept
        for the model defined by X and y'''
        x_min_index, x_max_index = np.argmin(X), np.argmax(X)
        x_min, x_max = X[x_min_index], X[x_max_index]
        y_min, y_max = y[x_min_index], y[x_max_index]
        
        self.coef_ = (y_max - y_min) / (x_max - x_min)
        self.intercept_ = y_min - self.coef_ * x_min
        
    def predict(self, x):
        return self.coef_ * x + self.intercept_


# If you did this problem correctly, the following code will create a linear model that can predict `mpg` from `disp`. 

# In[84]:


lin_mod1=MaxMinLinearRegression()
lin_mod1.fit(disp,mpg)


# To use your model to predict the `mpg` of an unknown car with `disp=200`, you would run this code:

# In[85]:


lin_mod1.predict(200)


# Run this to visualize your model's predictions, as compared to the actual:

# In[86]:


plt.scatter(disp,mpg)
plt.plot(disp,lin_mod1.predict(disp),'-r')
plt.xlabel('Displacement')
plt.ylabel('MPG')


# Calculate the RSS of `lin_mod1`.

# In[88]:


mpg_pred = np.array([lin_mod1.predict(x) for x in disp])
RSS1= np.sum((mpg - mpg_pred)**2)
RSS1


# Problem 2.
# 
# Create a class that finds a linear model describing the relationship between an array `X` of shape (num_observations,num_features) and a 1-dimensional array `y`, that minimizes the RSS. 

# In[89]:


class LinearRegression():
    def __init__(self):
        pass
    
    def fit(self, X, y): 
        '''Stores the slope and intercept
        for the model defined by X and y'''
        X = np.column_stack((np.ones(X.shape[0]), X)) 
        beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]
        
    def predict(self, x):
        '''x is expected to have shape 
        (num_test_obs, num_feats)'''
        x = np.atleast_2d(x) 
        x = np.column_stack((np.ones(x.shape[0]), x)) 
        return np.dot(x, np.concatenate(([self.intercept_], self.coef_)))


# To build a model to predict `mpg` from `disp`, we'll have to convert `disp` into an array of the correct shape:

# In[65]:


X=disp[:,np.newaxis]


# We now build the model, and fit it to the data:

# In[66]:


lin_mod2=LinearRegression()
lin_mod2.fit(X,mpg)


# To use this model to make a prediction on an unknown car with `disp=200`, you'll need to feed that value in as an array, rather than a single number.

# In[67]:


#lin_mod2.predict(200) won't work
lin_mod2.predict([200]) #Do this instead


# Run this code to visualize your model. Does it look better than the previous model?

# In[68]:


plt.scatter(disp,mpg)
plt.plot(disp,lin_mod2.predict(X),'-r')
plt.xlabel('Displacement')
plt.ylabel('MPG')


# Calculate the RSS of `lin_mod2`.

# In[78]:


pred = lin_mod2.predict(X)
RSS2= np.sum((mpg - pred) ** 2)
RSS2


# Problem 3.
# 
# We'll now bring in more features to see if we can predict `mpg` more accurately:

# In[70]:


wt=np.array(cars.weight)
gr=np.array(cars.gear_ratio)


# We build a feature matrix using displacement, weight, and gear ratio:

# In[71]:


DWG=np.array([disp,wt,gr]).T


# Now create a new linear model to predict `mpg` from the feature matrix `DWG`.

# In[72]:


lin_mod3 = LinearRegression()
lin_mod3.fit(DWG, mpg)


# Predict the `mpg` for a car with a displacement of 200, weight equal to 3000, and gear ratio 3.

# In[73]:


prediction=lin_mod3.predict(np.array([[200, 3000, 3]]))
prediction


# Calculate the RSS of `better_mod`.

# In[77]:


pred = lin_mod3.predict(DWG)
RSS3= np.sum((mpg - pred) ** 2)
RSS3


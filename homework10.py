#!/usr/bin/env python
# coding: utf-8

# **Homework 10**
# 

# In[19]:


import pandas as pd
import numpy as np


# *Problem 1.*
# 
# Let $f(x,y)=x^2+2xy+2y^2-4x-4y$.
# 
# Calculate  $\nabla f(x,y)$, the gradient of $f(x,y)$ on paper. (No need to turn this in, but you'll need it for the next parts of the problem.) In this problem you will use the gradient to find the minimum of $f(x,y)$. Do this first on paper by setting the gradient equal to $\langle 0,0 \rangle$ and solving for $x$ and $y$, so you can check that gradient descent is giving you the right answer.
# 

# Next, write a function fGD which implements gradient descent to find the minimum of $f(x,y)$. Your function should take in the following parameters:
# * `lr` (learning rate)
# * `max_iter` (maximum number of iterations)
# * `x_init` (initial value of x)
# * `y_init` (initial value of y)
# 
# Your function should return the final values of x and y

# In[20]:


def fGD(lr,max_iter,x_init,y_init):
    x=x_init
    y=y_init
    for _ in range(max_iter):
        grad_x = 2*x + 2*y - 4
        grad_y = 2*x + 4*y - 4
        x -= lr * grad_x
        y -= lr * grad_y
    return x,y


# Now check your answer by calling this function with a learning rate of 0.0001, max_iter of 10000, and inital values of 5 and 5 for `x` and `y`. Did your function come close to the correct answers?

# In[21]:


xmin1,ymin1=fGD(0.0001,10000,5,5) #Don't change this
xmin1,ymin1


# *Problem 2*
# 
# Write a class GDRegressor which implements gradient descent on MSE loss to fit an approximate linear model to a given data set.

# In[22]:


class GDRegressor():
    def __init__(self,learning_rate,max_iter):
        self.lr=learning_rate
        self.max_iter=max_iter

    def fit(self,X,y):
        self.coef=np.ones((X.shape[1],)) #Initial values
        self.intercept=1 #Initial value
        for i in range(self.max_iter):
            residuals = self.predict(X) - y            
            coef_grad=(2/X.shape[0])*np.dot(X.T, residuals)
            intercept_grad=(2/X.shape[0])*np.sum(residuals)
            self.coef-=self.lr*coef_grad
            self.intercept-=self.lr*intercept_grad

    def predict(self,X):
        return np.dot(X, self.coef)+self.intercept


# You can test your code here. Is the result close to what you would expect?

# In[23]:


x=np.arange(10)
y=3*x+2
X=x[:,np.newaxis] #Converts shape to (10,1)
lin_mod=GDRegressor(.01,2000)
lin_mod.fit(X,y)
lin_mod.coef, lin_mod.intercept


# We now try your new class on the `disp` vs `mpg` problem from previous assignments. Let's bring those data sets back:

# In[24]:


cars=pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/auto.csv')
disp=np.array(cars.displacement)
mpg=np.array(cars.mpg)

index=np.argsort(disp)
disp=disp[index]
mpg=mpg[index]


# Gradient descent works best with scaled data, so we'll need to import a `StandardScalar` class from sklearn:

# In[25]:


from sklearn.preprocessing import StandardScaler


# This class works almost exactly the same as the one you wrote in previous assignments, except that it expects a 2D-array, even when you have one column of data. To fix this, we reshape our data:

# In[26]:


disp=disp[:,np.newaxis]


# Now, fit a `StandardScaler` object to `disp` and transform it:

# In[27]:


scaler=StandardScaler()
scaled_disp=scaler.fit_transform(disp)


# Create a new `GDRegressor` object called `mpg_mod`. Use a learning rate of 0.1 and a `max_iter` of 1000. Then, fit it to `scaled_disp` and `mpg`. (Remember to first reshape `scaled_disp` appropriately).

# In[28]:


mpg_mod=GDRegressor(0.1,1000)
scaled_disp=scaled_disp.reshape(-1, 1)
mpg_mod.fit(scaled_disp, mpg)


# Check the RSS of your model, and compare your answer to the RSS of the model you found by the normal equations in Homework 7.

# In[29]:


mpg_pred=mpg_mod.predict(scaled_disp)
RSS=np.sum((mpg - mpg_pred) ** 2)
RSS


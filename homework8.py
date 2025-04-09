#!/usr/bin/env python
# coding: utf-8

# **Homework 8**

# In[49]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# We will continue with the displacement `disp` and `mpg` columns of the `cars` dataset, as in the last assignment. (This time we'll sort these by `disp` to make visualization easier later.)

# In[50]:


cars=pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/auto.csv')
disp=np.array(cars.displacement)
mpg=np.array(cars.mpg)

index=np.argsort(disp)
disp=disp[index]
mpg=mpg[index]


# In this assignment we will use feature engineering to create polynomial models for predicting `mpg` from `disp`. However, before we start looking at powers of `disp`, we'll need to scale this array to have mean 0 and standard deviation 1 (otherwise, the powers of the entries in `disp` will get too large). To the end, create a `StandardScaler` class that remembers the mean and standard deviation of each column of a feature matrix, and can scale and unscale datasets using these numbers.

# In[51]:


class StandardScaler():
    def __init__(self):
        pass

    def fit(self,X):
        self.mean = np.mean(X)
        self.std = np.std(X)

    def transform(self,X):
        return (X - self.mean) / self.std

    def inverse_transform(self,X):
        return (X*self.std) + self.mean


# Create a `StandardScaler` object, fit it to `disp`, and then create a new array called `scaled_disp`.

# In[52]:


disp_scaler=StandardScaler()
disp_scaler.fit(disp)
scaled_disp=disp_scaler.transform(disp)


# In the previous assignment you built a Linear Regression class, identical to the one packeged with sklearn. We'll import this here:

# In[53]:


from sklearn.linear_model import LinearRegression


# To create a higher order polynomial model, you'll have to first create a feature matrix with higher powers of `disp`. Create a class that does this for you for any input array `X`. The `fit_transform` method of this class will return a matrix whose first column is a column of ones (if `include_bias=True`), next column is `X`, next is `X**2`, etc.

# In[70]:


class PolynomialFeatures():
    def __init__(self,degree,include_bias=False):
        self.degree=degree
        self.include_bias=include_bias

    def fit_transform(self, X):
        X = np.asarray(X).reshape(-1, 1)  
        exponents = np.arange(1, self.degree + 1) 
        features = np.power(X, exponents) 
        if self.include_bias:
            features = np.hstack((np.ones((X.shape[0], 1)), features)) 
        return features


# Now, for example, if you wanted to create a matrix whose first column is `[0,1,2,3]` and second column is those values squared, you would do this:

# In[71]:


quad=PolynomialFeatures(2)
quad.fit_transform(np.array([0,1,2,3]))


# Generate a matrix whose columns are `scaled_disp` and `scaled_disp**2`.

# In[72]:


scaled_disp2=quad.fit_transform(scaled_disp)


# Create a quadratic model to predict `mpg` from `scaled_disp` by creating a linear model to predict `mpg` from both `scaled_disp` and `scaled_disp**2` (*i.e.* from `scaled_disp2`).

# In[73]:


quadratic_mod=LinearRegression()
quadratic_mod.fit(scaled_disp2, mpg)


# Now, apply this model to `scaled_disp2` to create an array of predictions.

# In[68]:


quad_preds=quadratic_mod.predict(scaled_disp2)


# Now visualize it:

# In[69]:


plt.scatter(disp,mpg)
plt.plot(disp,quad_preds,'-r')
plt.xlabel('Displacement')
plt.ylabel('MPG')


# Calculate the RSS of ```quadratic_mod```.

# In[29]:


RSSquad=np.sum((mpg-quad_preds)**2)
RSSquad


# Now create a cubic model of `mpg` vs `scaled_disp`, visualize it, and calculate its RSS.

# In[30]:


cubic = PolynomialFeatures(degree=3, include_bias=False)
scaled_disp3 = cubic.fit_transform(scaled_disp)
cubic_mod=LinearRegression()
cubic_mod.fit(scaled_disp3, mpg)
cubic_preds=cubic_mod.predict(scaled_disp3)


# In[31]:


plt.scatter(disp,mpg)
plt.plot(disp,cubic_preds,'-r')
plt.xlabel('Displacement')
plt.ylabel('MPG')


# In[32]:


RSScubic=np.sum((mpg-cubic_preds)**2)
RSScubic


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# **Homework 12**

# In[91]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Add the code to the SGDRegressor class below to allow for variable amount of L1 and L2 regularization.

# In[92]:


class SGDRegressor():
    def __init__(self,learning_rate, max_iter, batch_size, penalty='l2', alpha=0.0001):
        self.lr=learning_rate
        self.max_iter=max_iter #number of epochs
        self.batch_size=batch_size
        self.penalty=penalty #either 'l1' or 'l2'
        self.alpha=alpha #amount of regularization to apply
        
    def fit(self,X,y):
        self.coef=np.ones((X.shape[1],)) #Initial values
        self.intercept=1 #Initial value
        if self.penalty=='l1':
            penalty_grad=lambda x: self.alpha*np.sign(x) 
        elif self.penalty=='l2':
            penalty_grad=lambda x: self.alpha*x
        else:
            penalty_grad=lambda x: 0
        indices=np.arange(len(X))
        for i in range(self.max_iter):
            np.random.seed(i) #Just so everyone gets the same answer!
            np.random.shuffle(indices)
            X_shuffle=X[indices] 
            y_shuffle=y[indices] 
            for j in range(0,len(X),self.batch_size):
                X_batch=X_shuffle[j:j+self.batch_size]
                y_batch=y_shuffle[j:j+self.batch_size]
                residuals=self.predict(X_batch)-y_batch
                coef_grad=(X_batch.T)@residuals/len(X_batch)
                intercept_grad=np.mean(residuals)
                self.coef -= self.lr * coef_grad + penalty_grad(self.coef)
                self.intercept-=self.lr*intercept_grad+penalty_grad(self.intercept)
            
    def predict(self,X):
        return X@self.coef+self.intercept
    
    def mse(self,X,y): #Not a sklearn method, but added here for convenience
        return ((self.predict(X)-y)**2).mean()


# We now bring back some familiar classes:

# In[93]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split


# We now use L2 regularization to address overfitting of a polynomial model. To begin, we generate and visualize some data:

# In[94]:


#Generate Data
np.random.seed(4)
x=np.arange(0,1,1/16)
randoms=np.random.rand(16)
y=2*x+1+randoms
plt.scatter(x,y)


# Pre-process the data as follows:
# 1. Do a 75/25 train/test split
# 2. Create matrices from xtrain and xtest by adding columns corresponding to powers one through twelve of the original column. 

# In[95]:


#Process Data:
xtrain,xtest,ytrain,ytest= train_test_split(x.reshape(-1, 1), y, test_size=0.25, random_state=None)
poly= PolynomialFeatures(degree=12, include_bias=False)
deg12xtrain= poly.fit_transform(xtrain)
deg12xtest= poly.transform(xtest)


# Create an SGD model with no regularization. Use a learning rate of 0.1, max_iter of 10000, and batch_size of 16. Then, fit it to `deg12xtrain` and `ytrain`.

# In[96]:


noreg = SGDRegressor(learning_rate=0.1, max_iter=10000, batch_size=16, alpha=0)
noreg.fit(deg12xtrain, ytrain)


# Evaluate the MSE of your model on the train set:

# In[97]:


noreg_mse_train=noreg.mse(deg12xtrain, ytrain)
noreg_mse_train


# Evaluate the MSE of your model on the test set:

# In[98]:


noreg_mse_test=noreg.mse(deg12xtest, ytest)
noreg_mse_test


# Create a similar model, but with 0.01 of L2 regularization.

# In[99]:


regmodel = SGDRegressor(learning_rate=0.1, max_iter=10000, batch_size=16, penalty='l2', alpha=0.01)
regmodel.fit(deg12xtrain, ytrain)


# Evaluate the MSE of your new model on the train set. It will be higher than before, indicating a model that is not as good on the data it was trained on.

# In[100]:


reg_mse_train=regmodel.mse(deg12xtrain, ytrain)
reg_mse_train


# Evaluate the MSE of your new model on the test set. It should be lower than before, indicating a better (less overfit) model.

# In[101]:


reg_mse_test=regmodel.mse(deg12xtest, ytest)
reg_mse_test


# 
# 
# ---
# 
# You will now explore the use of L1 regularization to do feature selection on the classic California housing dataset. Import it here:
# 

# In[102]:


from sklearn.datasets import fetch_california_housing
ca=fetch_california_housing(as_frame=True).frame
ca.head(3)


# Check the shape to see how many observations and features you'll be dealing with:

# In[103]:


ca.shape


# Our goal is to find a model which predicts the MedHouseVal variable as accurately as possible, using as few of the other features as possible. 
# 
# To start, we'll create a feature matrix X containing all of the columns besides MedHouseVal, and a target vector y containing the entries in MedHouseVal.

# In[104]:


X=np.array(ca.loc[:,'MedInc':'Longitude'])
y=np.array(ca['MedHouseVal'])


# Now, do the following:
# 1. 80/20 Train-Test split.
# 2. Scale the training and testing feature matrices appropriately.
# 3. Fit a linear model using SGD on the train data with 1000 epochs, batch sizes of 5000, a learning rate of 0.01, and 0.001 of L1 regularization.
# 4. List the coefficients of your model.
# 5. Calculate the MSE on the test data.

# In[110]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)


# In[111]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[114]:


ca_mod= SGDRegressor(learning_rate=0.01, max_iter=1000, batch_size=5000, penalty='l1', alpha=0.001)
ca_mod.fit(X_train_scaled, y_train)


# In[115]:


ca_mod_coefficients=ca_mod.coef
ca_mod_coefficients


# In[116]:


ca_mod_mse=ca_mod.mse(X_test_scaled, y_test)
ca_mod_mse


# Now, do the following:
# 1. Identify the two coefficients with the largest absolute value 
# 3. Create new feature matrices X_train_small and X_test_small from X_train_scaled and X_test_scaled using only those columns corresponding to the two coefficients that have the largest absolute value.
# 4. Create a new SGD model, with the same parameters as the previous one, but fit to X_train_small and y_train.
# 5. Check the MSE of your new model and compare it to the MSE of the previous one.

# In[118]:


two_largest= np.argsort(np.abs(ca_mod.coef))[-2:]
X_train_small = X_train_scaled[:, two_largest]
X_test_small = X_test_scaled[:, two_largest]


# In[119]:


ca_mod_small= SGDRegressor(learning_rate=0.01, max_iter=1000, batch_size=5000, penalty='l1', alpha=0.001)
ca_mod_small.fit(X_train_small, y_train)
ca_mod_small_mse=ca_mod_small.mse(X_test_small, y_test)
ca_mod_small_mse


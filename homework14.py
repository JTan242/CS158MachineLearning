#!/usr/bin/env python
# coding: utf-8

# **Homework 14**

# In[8]:


import numpy as np
import pandas as pd


# *Background*
# 
# Recall from the last assignment that Logistic Regression was a way to leverage linear regression to perform binary classification. Here we make further modifications to perform multi-class classification. This is  called *Softmax Regression*. 
# 
# As in Logistic Regression, the model we create will predict probabilities, and those probabilities will determine class predictions. The predictied probabilities will be a matrix of shape (n,k), where n is the number of observations and k is the number of classes. 
# 
# The input to the model is a feature matrix $X$ of shape (n,m), where as usual n is the number of observations, and m is the number of features. The model is then defined by a coefficient matrix of shape (m,k) and an intercept array of shape (k,). 
# 
# The first step in finding a matrix of predicted probabilites is to compute the matrix $$t=X \cdot coef + intercept$$ (similar to linear regression, but note the different shapes!). The probability matrix $p$ is obtained from this matrix $t$ by the softmax function:
# $$p_i^j=\frac{e^{t_i^j}}{\sum \limits _{j=1} ^k e^{t_i^j}}.$$
# 
# To find the coefficient matrix and intercept array we must first preprocess the target array by doing a *one-hot encoding*. That is, we take a vector $y$ of $n$ entries, where each entry is one of $k$ different classes, and convert it to a matrix $Y$ of shape (n,k) whose entries are all 1's and 0's. In $Y$, a 1 in row i, column j indicates that $y_i$ is in category $j$. 
# 
# Our goal is to find the coefficent matrix and intercept array so that the probability $p_i^j$ is close to one if $Y_i^j=1$, and close to zero otherwise. In Softmax regression we accomplish this by using gradient descent to minimize the *categorical cross entropy loss*:
# $$CE=-\frac{1}{n}\sum \limits _{i,j} Y_i^j \mbox{Log}(p_i^j)$$
# 
# With these definitions, the forumlas for the gradient calculation and coefficient/intercept updates are the same as in Logistic Regression. This should not be surprising, as binary Softmax Regression is mathematically identical to Logistic Regression.
# 

# Before we define our `SoftmaxRegression()` class, we'll need to create a class that performs the one-hot encoding:

# In[9]:


class OneHotEncoder():
    def __init__(self):
        pass
    
    def fit(self,y):
        self.categories= np.unique(y)
        self.n_features_in= len(self.categories)
        
    def transform(self,y):
        Y = np.zeros((len(y), self.n_features_in)) 
        for i in range(len(y)):
            category_index = np.where(self.categories == y[i])[0][0] 
            Y[i, category_index] = 1
        return Y
    
    def fit_transform(self,y):
        '''Convenience method that applies fit and then 
        immediately transforms'''
        self.fit(y)
        return self.transform(y)


# Test your class here:

# In[10]:


y=np.array(['a','b','a','c','b','b'])
encoder=OneHotEncoder()
encoder.fit_transform(y)


# Defining `OneHotEncoder()` as a class will allow us to fit our encoder to a train set but apply it to a test set:

# In[11]:


z=np.array(['b','c','b'])
encoder.transform(z)


# Note that this is a different result than if we had done `encoder.fit_transform(z)`.

# We are now ready to create our `SoftmaxRegression()` class. Below is most of the code from the `LogisticRegression()` class. Modify it appropriately, where indicated. 

# In[18]:


class SoftmaxRegression():
    def __init__(self,learning_rate, max_iter, batch_size, penalty='l2', alpha=0.0001):
        self.lr=learning_rate
        self.max_iter=max_iter 
        self.batch_size=batch_size
        self.penalty=penalty 
        self.alpha=alpha 
        self.encoder=OneHotEncoder() 
        
    def fit(self,X,y):
        Y=self.encoder.fit_transform(y)
        self.coef=np.ones((X.shape[1], Y.shape[1])) 
        self.intercept=np.ones((Y.shape[1],)) 
        if self.penalty=='l1':
            penalty_grad=lambda x:2*(x>0)-1
        elif self.penalty=='l2':
            penalty_grad=lambda x:x
        else:
            penalty_grad= lambda x: 0 
        indices=np.arange(len(X))
        for i in range(self.max_iter):
            np.random.seed(i) 
            np.random.shuffle(indices)
            X_shuffle=X[indices] 
            Y_shuffle=Y[indices] 
            for j in range(0,len(X),self.batch_size):
                X_batch=X_shuffle[j:j+self.batch_size]
                Y_batch=Y_shuffle[j:j+self.batch_size] #Note that we're using Y here, not y
                residuals=self.predict_proba(X_batch)-Y_batch 
                coef_grad=(X_batch.T)@residuals/len(X_batch)
                intercept_grad=np.mean(residuals)
                self.coef-=self.lr*coef_grad+self.alpha*penalty_grad(self.coef)
                self.intercept-=self.lr*intercept_grad+self.alpha*penalty_grad(self.intercept)
            
    def predict_proba(self,X):
        '''returns the matrix of predicted probabilites'''
        t = X @ self.coef + self.intercept
        exponent_t = np.exp(t - np.max(t, axis=1, keepdims=True)) 
        return exponent_t / np.sum(exponent_t, axis=1, keepdims=True) 
    
    def predict(self,X):
        '''returns a prediction, for each observation in X, 
        of one category.'''
        probs = self.predict_proba(X)
        return self.encoder.categories[np.argmax(probs, axis=1)]
    
    def score(self,X,y): 
        '''returns accuracy of the model'''
        return (self.predict(X)==y).mean()
    
    def CEloss(self,X,y): #Not a sklearn method!
        '''returns the Categorical Cross Entropy loss'''
        Y = self.encoder.transform(y)
        probs = self.predict_proba(X)
        return -np.mean(np.sum(Y * np.log(probs + 1e-9), axis=1))


# Note that we have departed here from `sklearn` syntax. To perform Softmax Regression with that package, call the `LogisticRegression()` class and set `multi_class=multinomial`.

# We'll now test your `SoftmaxRegression()` class on some real data. To keep things realistic, we should first do a train/test split, so we'll need to bring back this function:

# In[19]:


def TrainTestSplit(x,y,p,seed=4):
    '''Splits datasets x and y into train and test sets
    p is the fraction going to train'''
    rng = np.random.default_rng(seed=seed)#Don't change seed!
    size=len(x)
    train_size=int(p*size)
    train_mask=np.zeros(size,dtype=bool)
    train_indices=rng.choice(size, train_size, replace=False)  
    train_mask[train_indices]=True
    test_mask=~train_mask
    x_train=x[train_mask]
    x_test=x[test_mask]
    y_train=y[train_mask]
    y_test=y[test_mask]
    return x_train,x_test,y_train,y_test


# We'll now test your `SoftmaxRegression()` class on the iris dataset:

# In[20]:


iris=(pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/datasets/iris.csv',index_col=0))
X=np.array(iris.loc[:,'Sepal.Length':'Petal.Width'])
y=np.array(iris['Species'])


# Again, do an 80/20 train/test split. As this dataset is much smaller, your test set will only contain 30 observations. 

# In[21]:


Xtrain,Xtest,ytrain,ytest=TrainTestSplit(X, y, 0.8)


# Create a Softmax Regression object with a learning rate of 0.01, which trains for 1000 epochs with batches of size of 50, and no regularization. Then, fit it to `Xtrain` and `ytrain`. 

# In[22]:


mod=SoftmaxRegression(learning_rate=0.01, max_iter=1000, batch_size=50, penalty=None)
mod.fit(Xtrain, ytrain)


# Check the accuracy on the test set:

# In[23]:


accuracy= mod.score(Xtest, ytest)
accuracy


# Multiplying this by 30 (the size of the test set) will reveal how many predictions were correct:

# In[24]:


accuracy*30


# List all 30 species that your model predicts from `Xtest`. (If you compare this to `ytest`, you would be able to determine which flower species the model had trouble with.)

# In[25]:


predictions= mod.predict(Xtest)
predictions


# Check the cross entropy loss on the test data. (This won't be very meaningful on its own, but would be useful in comparing the effects of different choices of `max_iter`, `batch_size`, learning rate, and amount of regularization.)

# In[26]:


loss= mod.CEloss(Xtest, ytest)
loss


# What does your model say that the probability of a flower with Sepal Length 4, Sepal Width 3, Petal Length 2, and Petal Width 1 is of being a setosa?

# In[27]:


setosa_prob= mod.predict_proba(np.array([[4, 3, 2, 1]]))[0, np.where(mod.encoder.categories == 'setosa')[0][0]]
setosa_prob


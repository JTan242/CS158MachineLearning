#!/usr/bin/env python
# coding: utf-8

# **Homework 4**

# In[49]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


# Let's start by importing a new dataset. You should read about it [here](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-dataset).

# In[50]:


wine=load_wine()
X=wine.data
y=wine.target

X.shape


# Notice that the feature matrix has 13 different columns, so its very hard to visualize the data. Let's use PCA:

# In[51]:


pca=PCA(n_components=3)
projectedX=pca.fit_transform(X)
plt.scatter(projectedX[:,0],projectedX[:,1],c=y)


# Observe that the different classes are fairly well clustered, making this a decent choice for KNN. However, in this assignment we'll try a decision tree classifier instead.
# 
# Begin by creating a function that takes a set of labels and determines its Gini index. (*Hint:* Numpy's `bincount` function is useful here.)

# In[62]:


def Gini(labels):
    '''Returns Gini index of a set of labels.'''
    if len(labels) == 0:
        return 0 
    
    unique_labels = np.max(labels) + 1  
    counts = np.bincount(labels, minlength=unique_labels)  
    probabilities = counts / np.sum(counts)  
    
    gini = 1.0 - np.sum(probabilities ** 2)
    
    return float(gini)


# Next, create a function that takes in a set of data `X`, a target set `y`, a column to use to split the data, and a threshold which deteremines the splitting. The function should return the Gini index of that split.

# In[53]:


def GiniSplit(X,y,split_feature,split_threshold):
    '''returns Gini index of a particular split of X'''
    #YOUR CODE HERE
    left_thresh = X[:, split_feature] < split_threshold
    right_thresh = X[:, split_feature] >= split_threshold
    
    left_y = y[left_thresh]
    right_y = y[right_thresh]
    
    # Compute Gini indices for left and right splits
    if len(left_y) == 0:
        gini_left = 0
    else:
        gini_left = Gini(left_y)
        
    if len(right_y) == 0:
        gini_right = 0
    else:
        gini_right = Gini(right_y)        
        
    
    # Compute weighted Gini index for the split
    total_samples = len(y)
    weighted_gini = ((len(left_y)/total_samples)*gini_left) + ((len(right_y)/total_samples)*gini_right)
    
    return weighted_gini


# Now create a function to determine the best value to split on. To do this, try splitting on every feature and the midpoints between distinct observations in `X`. (There are more efficient ways to do this, but you don't need to get fancy for this assignment.) You may use a nested for-loop to do this problem.

# In[54]:


def BestSplit(X,y):
    '''returns split_feature and split_threshold of split with lowest gini index'''
    num_obs,num_features=X.shape
    split_feature=0
    split_threshold=0
    minGini=1 #Largest possible Gini index
    for feat in range(num_features):
        vals=np.unique(X[:,feat]) #unique, sorted values of each feature
        thresholds=(vals[1:]+vals[:-1])/2 #Midpoints between successive values
        for thresh in thresholds:
            gini = GiniSplit(X, y, feat, thresh)
            if gini < minGini:
                minGini = gini
                split_feature = feat
                split_threshold = thresh
    return split_feature,split_threshold


# Finally, we are ready to create a class that will encode a node of a decision tree. The `__init__` method should create everything that defines a node. If this is a branching node, this will include which feature to split on, a splitting threshold, and two child nodes. If its a leaf node, then the only attribute we'll need is the label of that node, which predicts some class.
# 
# In either case, we'll also have to keep track of the depth of the node in the tree. A node is a leaf if either the maximum depth has been reached, or if the node is "pure" (i.e. all labels are the same).

# In[55]:


class Node():
    def __init__(self,X,y,depth,max_depth):
        self.leaf=(depth==max_depth or len(np.unique(y))==1)
        if self.leaf:
            self.label= np.bincount(y).argmax() 
        else:
            self.split_feature, self.split_threshold = BestSplit(X, y)
            left_thresh = X[:, self.split_feature] <= self.split_threshold
            right_thresh = ~left_thresh
            self.left = Node(X[left_thresh], y[left_thresh], depth + 1, max_depth)
            self.right = Node(X[right_thresh], y[right_thresh], depth + 1, max_depth)
    
    def predict(self,x):
        if self.leaf:
            return self.label
        else:
            if x[self.split_feature] <= self.split_threshold:
                return self.left.predict(x)
            else:
                return self.right.predict(x)


# To create a Decision Tree classifier, we'll make a new class that encodes the root node of a Tree. (The only thing that makes such a node special is that its depth is 0.) The syntax of this class will follow a common pattern in Machine Learning packages. Instantiating an object of the class only records hyper-parameters. The model itself doesn't exist until you call the `fit` method on a known Data matrix and correspondng Target array. Finally, the `predict` method is used to apply the model to a new data point and generate a prediction.  

# In[56]:


class DecisionTreeClassifier():
    def __init__(self,max_depth):
        self.max_depth=max_depth #record the only hyperparameter

    def fit(self,X,y):
        self.Node=Node(X,y,0,self.max_depth) #Define a depth 0 node from X,y

    def predict(self,x):
        return self.Node.predict(x)


# We are now ready to try out a decision tree classifier on the wine dataset. First, we'll have to do a Train/Test split, as in Homework 2. We'll use the sklearn `train_test_split` utility to do this, rather then the code from Homework 2.

# In[57]:


train_data, test_data, train_target, test_target = train_test_split(
    X, y, test_size=0.2, random_state=6)


# Now we are ready to create a Decision Tree, based on the training data:

# In[58]:


tree=DecisionTreeClassifier(5) #Create an empty decision tree with max_depth=5
tree.fit(train_data,train_target) #Fit the model to the data


# Let's test your classifier on one point in the test set:

# In[59]:


tree.predict(test_data[10])


# Compare your answer to the actual label. Did your classifier get it right?

# In[60]:


test_target[10]


# Now we'll check the accuracy (make sure you understand this code!):

# In[61]:


accuracy=(test_target==np.apply_along_axis(lambda x:tree.predict(x),1,test_data)).sum()/len(test_target)
accuracy


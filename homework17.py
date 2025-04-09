#!/usr/bin/env python
# coding: utf-8

# **Homework 17**
# 
# In this assignment, you'll add to [Andrej Karpathy's "micrograd"](https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py), a simple immplementation of an autograd engine for scalar-valued functions.  
# 

# In[1]:


import numpy as np


# Add the indicated methods to the following class definition.

# In[10]:


class tensor():
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=()):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __add__(self, other): #self + other
        out = tensor(self.data + other.data, (self, other))
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other): #self * other
        out = tensor(self.data * other.data, (self, other))
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other): # self**n
        out = tensor(self.data**other, (self,))
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def exp(self): # self.exp()
        out = tensor(np.exp(self.data), (self,))
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self): # self.log()
        out = tensor(np.log(self.data), (self,))
        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out

    def sin(self): # self.sin()
        out = tensor(np.sin(self.data), (self,))
        def _backward():
            self.grad += np.cos(self.data) * out.grad
        out._backward = _backward
        return out

    def cos(self): # self.cos()
        out = tensor(np.cos(self.data), (self,))
        def _backward():
            self.grad += -np.sin(self.data) * out.grad
        out._backward = _backward
        return out

    def __neg__(self): # -self
        return self * tensor(-1)

    def __sub__(self, other): # self - other
        return self + (-other)

    def __truediv__(self, other): # self/other
        out = tensor(self.data / other.data, (self, other))
        def _backward():
            self.grad += (1 / other.data) * out.grad
            other.grad += (-self.data / (other.data ** 2)) * out.grad
        out._backward = _backward
        return out

    def backward(self): #Implementation of backprop

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()



# Let's make sure it works. First, we'll run this test:

# In[11]:


a=tensor(2)
b=tensor(3)
d=a*(b**2)
d.backward()
a.grad, b.grad


# Check (by hand) that this is right!

# Next, observe what happens when we do this:

# In[12]:


d.backward()
a.grad, b.grad


# We now get the wrong answer because autograd engines often *accumulate* gradients, which causes problems when you differentiate more than once. To get the correct answer again, you have to manually zero out all gradients before recalculating:

# In[13]:


#manually zero out gradients w.r.t. params:
a.grad=0
b.grad=0

#rebuild computation graph:
d=a*(b**2)

#Calculate gradients:
d.backward()
a.grad, b.grad


# When we get to PyTorch in the next assignment you'll see this is a common workflow: Always zero out all gradients before calling the backward() method (especially when re-using tensors)!

# Problem 1. Calculate the partial derivatives of the function $f(x,y)=\frac{x^2+ e^y}{sin(xy)}$ where $x=-2$ and $y=2$.

# In[14]:


x = tensor(-2.0)
y = tensor(2.0)

numerator = (x ** 2) + y.exp()
denominator = (x * y).sin()
f = ((x ** 2) + y.exp()) / ((x * y).sin())
f.backward()

f_x=x.grad
f_y=y.grad
f_x, f_y


# Problem 2. Calculate the derviatve of $h(u)=\log(\sqrt u)$ where $u=4$.

# In[15]:


u = tensor(4.0)
h = u.log() * tensor(0.5)
h.backward()

h_u=u.grad
h_u


# Problem 3. Use gradient descent to find the values of $s$ and $t$ where there is a local minimum of $g(s,t)=\frac{e^s}{s}+(\log t)^2$. Start with $s=2$ and $t=3$. Use a learning rate of 0.001 and 10000 steps.

# In[16]:


s=tensor(2)
t=tensor(3)
lr = 0.001
for i in range(10000):
    s.grad = 0
    t.grad = 0
    g = s.exp() / s + (t.log() ** 2)
    g.backward()
    s.data -= lr * s.grad
    t.data -= lr * t.grad
    
final_s=s.data
final_t=t.data
final_s,final_t


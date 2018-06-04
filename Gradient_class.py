#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 09:22:15 2018

@author: jeMATHfischer

The parameter layer gives the number of layers and neurons the corresponding number of neurons. neurons should be a list or np.array of length layer
data gives the data input to be place in various positions along the weights.
Consider the weight matrices mxn as vector of length m*n. Then the partial derivatives along line k just give a vector with n entries, which is just the data.  
"""
#
#class ffnet_differential(object):
#    
#    def __init__(self, data, layer, neurons):
#        self.data = data
#        self.layer = layer
#        self.neurons = neurons
#        
#    
#    def grad(output1, output2, w2, data):
#        remain_len = dim_in*hidden_dim - len(data)
#    
#    return s_diff(output2)*(w2[0]*s_diff(output1[0])*np.append(np.array(data),np.zeros(remain_len)) + w2[1]*s_diff(output1[1])*np.append(np.zeros(remain_len), np.array(data)))     
#
#M = np.zeros((neurons[1]*neurons[2], neurons[1]))
#
#for i in range(neurons[2]):
#    for k in range(neurons[1]):
#        
#
#    dot(w2*s_diff(output), )
#    
#    
#    

from scipy.linalg import block_diag
import numpy as np

def grad(output1, output2, w2, input_data):
    np.dot(w2* s_diff(output1), input_data)
    return s_diff(output2)*np.dot(w2* s_diff(output1), input_data)

def conc(a,n,m = np.array([1])):
    if n == 1:
        m = np.delete(m,0,0)
        m = np.delete(m,0,1)
        return m
    else: 
        return conc(a,n-1,block_diag(m, a)) 

input_data = conc(input_data, hidden_dim)

np.dot(w2* s_diff(output1), input_data)

M = 
a a a 0 0 0
0 0 0 a a a


data 0 0
0 data 0
0 0 data

3 2 1 layer 

3xk weight matrix for first level

n_21 = w_11 x_1 + w_12 x_2 + w_13 x_3
.
.
.
n_2k = w_k1 x_1 + x_k2 x_2 + w_k3 x_3




#%%
from scipy.linalg import block_diag
import numpy as np

a = np.array([[1,0,0],[0,1,0]])
b = np.array([])


def conc(a,n,m = np.array([1])):
    if n == 1:
        m = np.delete(m,0,0)
        m = np.delete(m,0,1)
        return block_diag(m, a)
    else: 
        return conc(a,n-1,block_diag(m, a)) 

c = conc(a[0,:],3)
print(c)
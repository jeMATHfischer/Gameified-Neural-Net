#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 09:22:15 2018

@author: jeMATHfischer

The parameter layer gives the number of layers and neurons the corresponding number of neurons. neurons should be a list or np.array of length layer
data gives the data input to be place in various positions along the weights.
Consider the weight matrices mxn as vector of length m*n. Then the partial derivatives along line k just give a vector with n entries, which is just the data.  
"""

class ffnet_differential(object):
    
    def __init__(self, data, layer, neurons):
        self.data = data
        self.layer = layer
        self.neurons = neurons
        
    
    def grad(output1, output2, w2, data):
        remain_len = dim_in*hidden_dim - len(data)
    
    return s_diff(output2)*(w2[0]*s_diff(output1[0])*np.append(np.array(data),np.zeros(remain_len)) + w2[1]*s_diff(output1[1])*np.append(np.zeros(remain_len), np.array(data)))     

M = np.zeros((neurons[1]*neurons[2], neurons[1]))

for i in range(neurons[2]):
    for k in range(neurons[1]):
        

    dot(w2*s_diff(output), )
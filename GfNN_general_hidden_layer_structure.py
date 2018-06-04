#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 20:52:48 2018

@author: jens
"""

import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

np.random.seed(41)

# pseudo data 
x = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
# labels
y = np.array([[0,0,1,1]]).T

test = np.array([[1,0,0],[0,0,0],[1,1,0], [0,1,0]])
# If there are more 0 then net tends to advice to pick zero no matter what the characteristics of the data

alpha, hidden_dim = (2,4)

def s_diff(value):
    return value*(1-value)

dim_in = x.shape[1]

def grad(output1, output2, w2, input_data):
    np.dot(w2* s_diff(output1), input_data)
    return s_diff(output2)*np.dot(np.reshape(w2, (1,-1))*s_diff(output1), input_data)

def conc(a,n,m = np.array([1])):
    if n == 1:
        m = np.delete(m,0,0)
        m = np.delete(m,0,1)
        return block_diag(m, a)
    else: 
        return conc(a,n-1,block_diag(m, a)) 

nonlinearity_1 = lambda s: 1/(1 + np.exp(-s))

count = 0
tries = 100
#
for p in range(tries):

    synapse_0 = 2*np.random.random((dim_in,hidden_dim)) -1 
    synapse_1 = 2*np.random.random((hidden_dim, 1)) - 1
    
    
    for l in range(100):        
        for i in range(len(y)):
            for l in range(10):    
                layer_1 = nonlinearity_1(np.dot(x[i,:], synapse_0))
                layer_2 = nonlinearity_1(np.dot(layer_1, synapse_1))
                # meaning of delta objects?
                synapse_0 = np.reshape(synapse_0, (-1,1))
                input_data = conc(x[i,:], hidden_dim)
                synapse_0 = synapse_0 - alpha*np.reshape(-2*(y[i]-layer_2)*grad(layer_1, layer_2, synapse_1, input_data), (hidden_dim*dim_in,-1))
                synapse_0 = np.reshape(synapse_0, (dim_in,-1))
            
                layer_1 = nonlinearity_1(np.dot(x[i,:], synapse_0))
                layer_2 = nonlinearity_1(np.dot(layer_1, synapse_1))
                layer_2_delta = 2*(layer_2 - y[i])*(layer_2*(1-layer_2))
                synapse_1 -= alpha*np.reshape(layer_1, (hidden_dim,-1))*(layer_2_delta)
            
    
    np.random.shuffle(test)
    layer_1_test = nonlinearity_1(np.dot(test[0,:], synapse_0))
    layer_2_test = nonlinearity_1(np.dot(layer_1_test, synapse_1))
    print('try {} gives estimate {} for {}'.format(p,layer_2_test, test[0,0]))
    if test[0,0] == 1:
        count += int(0.6 < layer_2_test)
    else:
        count += int(0.4 > layer_2_test)

print(count/tries)        
        
# ----------------------- Testing area
for i in range(len(y)):
    layer_1_test = nonlinearity_1(np.dot(x[i,:], synapse_0))
    layer_2_test = nonlinearity_1(np.dot(layer_1_test, synapse_1))
    print('y: ' + str(y[i]))
    print('predict: ' + str(int(0.5 < layer_2_test)))       
    

layer_1_test = nonlinearity_1(np.dot(np.array([1,0,0]), synapse_0))
layer_2_test = nonlinearity_1(np.dot(layer_1_test, synapse_1))
print('y: ' + str(1))
print('predict: ' + str(int(0.6 < layer_2_test)))       
print('predict: ' + str(layer_2_test))       

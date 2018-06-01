#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 07:40:51 2018

@author: jens
"""

import numpy as np
alphas = 0.01#[0.001,0.01,0.1,1,10,100,1000]
 
# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output
# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)

x = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])
y = np.array([[0],
              [1],
              [1],
              [0]])
    
#for alpha in alphas:
 #   print( "\nTraining With Alpha:" + str(alpha))

alpha = 0.01

np.random.seed(1)
# randomly initialize our weights with mean 0
#synapse_0 = 2*np.random.random((3,2)) - 1
#synapse_1 = 2*np.random.random((2,1)) - 1

count = 0
    
for p in range(100):
    synapse_0 = 2*np.random.random((3,2)) - 1
    synapse_1 = 2*np.random.random((2,1)) - 1

    for j in range(1000):
        layer_0 = x
        layer_1 = sigmoid(np.dot(layer_0,synapse_0))
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))
        layer_2_error = layer_2 - y
     
        layer_2_delta = layer_2_error*sigmoid_output_to_derivative(layer_2)
        layer_1_error = layer_2_delta.dot(synapse_1.T)
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
    
        synapse_1 -= alpha * (layer_1.T.dot(layer_2_delta))
        synapse_0 -= alpha * (layer_0.T.dot(layer_1_delta))
        
    layer_1_test = sigmoid(np.dot(np.array([1,0,0]), synapse_0))
    layer_2_test = sigmoid(np.dot(layer_1_test, synapse_1))
    count += int(0.6 < layer_2_test)
        
print(count/100)
#    
#print('synapse_0:')
#print(synapse_0)
#print('synapse_1:')
#print(synapse_1)
#
#for i in range(len(y)):
#    layer_1_test = sigmoid(np.dot(x[i,:], synapse_0))
#    layer_2_test = sigmoid(np.dot(layer_1_test, synapse_1))
#    print('y: ' + str(y[i]))
#    print('predict: ' + str(int(0.5 < layer_2_test)))       
#    
#
#layer_1_test = sigmoid(np.dot(np.array([1,0,1]), synapse_0))
#layer_2_test = sigmoid(np.dot(layer_1_test, synapse_1))
#print('y: ' + str(1))
#print('predict: ' + str(int(0.5 < layer_2_test)))       

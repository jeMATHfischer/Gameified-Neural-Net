import numpy as np
import matplotlib.pyplot as plt

# pseudo data 
x = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
# labels
y = np.array([[0,0,1,1]]).T

# If there are more 0 then net tends to advice to pick zero no matter what the characteristics of the data

alpha, hidden_dim = (0.5,2)
test = np.array([[0,0]])

def s_diff(value):
    return value*(1-value)

dim_in = x.shape[1]

def grad(output1, output2, w2, data):
    remain_len = dim_in*hidden_dim - len(data)
    return s_diff(output2)*(w2[0]*s_diff(output1[0])*np.append(np.array(data),np.zeros(remain_len)) + w2[1]*s_diff(output1[1])*np.append(np.zeros(remain_len), np.array(data)))     


#building synapses with values in [-1,1]
#synapse_0 = 2*np.random.random((dim_in,hidden_dim)) -1 
#synapse_1 = 2*np.random.random((hidden_dim, 1)) - 1

#
#synapse_0 = -np.zeros((dim_in,hidden_dim))
#synapse_1 = -np.zeros((hidden_dim, 1))
#
#print('starting synapses')
#print(synapse_0)
#print(synapse_1)

nonlinearity_1 = lambda s: 1/(1 + np.exp(-s))

count = 0

for p in range(100):

    synapse_0 = 2*np.random.random((dim_in,hidden_dim)) -1 
    synapse_1 = 2*np.random.random((hidden_dim, 1)) - 1
    
    for l in range(10000):        
        for i in range(len(y)):    
            layer_1 = nonlinearity_1(np.dot(x[i,:], synapse_0))
            layer_2 = nonlinearity_1(np.dot(layer_1, synapse_1))
            # meaning of delta objects?
            synapse_0 = np.reshape(synapse_0, (-1,1))
            synapse_0 = synapse_0 - alpha*np.reshape(-2*(y[i]-layer_2)*grad(layer_1, layer_2, synapse_1, x[i,:]), (hidden_dim*dim_in,-1))
            synapse_0 = np.reshape(synapse_0, (dim_in,-1))
        
            layer_1 = nonlinearity_1(np.dot(x[i,:], synapse_0))
            layer_2 = nonlinearity_1(np.dot(layer_1, synapse_1))
            layer_2_delta = 2*(layer_2 - y[i])*(layer_2*(1-layer_2))
            synapse_1 -= alpha*np.reshape(layer_1, (2,-1))*(layer_2_delta)
            
    layer_1_test = nonlinearity_1(np.dot(np.array([1,0,0]), synapse_0))
    layer_2_test = nonlinearity_1(np.dot(layer_1_test, synapse_1))
    count += int(0.6 < layer_2_test)
        
print(count/100)

#print('synapse_0:')
#print(synapse_0)
#print('synapse_1:')
#print(synapse_1)
#
#for i in range(len(y)):
#    layer_1_test = nonlinearity_1(np.dot(x[i,:], synapse_0))
#    layer_2_test = nonlinearity_1(np.dot(layer_1_test, synapse_1))
#    print('y: ' + str(y[i]))
#    print('predict: ' + str(int(0.5 < layer_2_test)))       
#    
#
#layer_1_test = nonlinearity_1(np.dot(np.array([1,0,0]), synapse_0))
#layer_2_test = nonlinearity_1(np.dot(layer_1_test, synapse_1))
#print('y: ' + str(1))
#print('predict: ' + str(int(0.6 < layer_2_test)))       
#print('predict: ' + str(layer_2_test))       

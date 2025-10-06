# Introduction to Machine Learning
# Jesús Urtasun Elizari: MRC LMS 2026
# Chapter 1: What is learning



# Exercise 1: 
# Implement a series of activation functions: ReLU, sigmoid, softmax
# Check calculation with tensorflow / pytorch implementation



# Import libraries ............................................................

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch




# Generate example input ......................................................

x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
print("\nExplore input:")
print("x: ", x)
print("x shape:", x.shape)
print("x format: ", type(x))



# Manual implementation .......................................................

# ReLU
def relu_manual(x):
    return np.maximum(0, x)

# Sigmoid
def sigmoid_manual(x):
    return 1 / (1 + np.exp(-x))

# Softmax
def softmax_manual(x):
    exps = np.exp(x - np.max(x))  # for numerical stability
    return exps / np.sum(exps)

# Check manual implementation
sigmoid_m = sigmoid_manual(x)
relu_m = relu_manual(x)
softmax_m = softmax_manual(x)
print("\nManual implementation:")
print("Sigmoid:", sigmoid_m)
print("ReLU:", relu_m)
print("Softmax:", softmax_m)



# TensorFlow implementation ...................................................

# Convert input to required format
x_tf = tf.constant(x, dtype = tf.float32)

# Check tensorflow implementation
sigmoid_tf = tf.nn.sigmoid(x_tf).numpy()
relu_tf = tf.nn.relu(x_tf).numpy()
softmax_tf = tf.nn.softmax(x_tf).numpy()
print("\nTensorFlow implementation:")
print("Sigmoid:", sigmoid_tf)
print("ReLU:", relu_tf)
print("Softmax:", softmax_tf)



# PyTorch implementation ......................................................

# Convert input to required format
x_torch = torch.tensor(x, dtype = torch.float32)

# Check pytorch implementation
sigmoid_torch = torch.sigmoid(x_torch).numpy()
relu_torch = torch.relu(x_torch).numpy()
softmax_torch = torch.softmax(x_torch, dim=0).numpy()
print("\nPyTorch implementation:")
print("Sigmoid:", sigmoid_torch)
print("ReLU:", relu_torch)
print("Softmax:", softmax_torch)



# Precision check .............................................................

print("\nPrecision check (Manual vs TensorFlow):")
print("Sigmoid close: ", np.allclose(sigmoid_m, sigmoid_tf))
print("ReLU close: ", np.allclose(relu_m, relu_tf))
print("Softmax close: ", np.allclose(softmax_m, softmax_tf))

print("\nCross-check (Manual vs PyTorch):")
print("Sigmoid close: ", np.allclose(sigmoid_m, sigmoid_torch))
print("ReLU close: ", np.allclose(relu_m, relu_torch))
print("Softmax close: ", np.allclose(softmax_m, softmax_torch))

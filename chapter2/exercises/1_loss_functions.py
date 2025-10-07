# Introduction to Machine Learning
# Jesús Urtasun Elizari: MRC LMS 2026
# Chapter 2: Basic mathematical intuitions



# Exercise 1: 
# Implement a series of loss functions: MSE, cross entropy, categorical cross entropy
# Check calculation with tensorflow / pytorch implementation



# Import libraries ............................................................

import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F



# Sample input and output .....................................................

y_true_cont = np.array([1.0, 2.0, 3.0])       # continuous for MSE/Chi2
y_pred_cont = np.array([1.1, 1.9, 3.2])

y_true_bin = np.array([1, 0, 1])              # binary labels
y_pred_bin = np.array([0.9, 0.2, 0.8])

y_true_cat = np.array([[1,0,0],[0,1,0],[0,0,1]])  # one-hot for categorical
y_pred_cat = np.array([[0.8,0.1,0.1],[0.1,0.7,0.2],[0.2,0.2,0.6]])



# Manual implementations ......................................................

def mse_manual(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def chi2_manual(y_true, y_pred):
    return np.sum((y_true - y_pred)**2 / (y_pred + 1e-7))

def binary_ce_manual(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1-1e-7)
    return -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))

def categorical_ce_manual(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1-1e-7)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))



# TensorFlow computations .....................................................

mse_tf = tf.keras.losses.MeanSquaredError()(y_true_cont, y_pred_cont).numpy()
chi2_tf = tf.reduce_sum((y_true_cont - y_pred_cont)**2 / (y_pred_cont + 1e-7)).numpy()
bce_tf = tf.keras.losses.BinaryCrossentropy()(y_true_bin, y_pred_bin).numpy()
cce_tf = tf.keras.losses.CategoricalCrossentropy()(y_true_cat, y_pred_cat).numpy()



# PyTorch computations ........................................................

y_true_cont_t = torch.tensor(y_true_cont, dtype = torch.float32)
y_pred_cont_t = torch.tensor(y_pred_cont, dtype = torch.float32)
y_true_bin_t = torch.tensor(y_true_bin, dtype = torch.float32)
y_pred_bin_t = torch.tensor(y_pred_bin, dtype = torch.float32)
y_true_cat_t = torch.tensor(y_true_cat, dtype = torch.float32)
y_pred_cat_t = torch.tensor(y_pred_cat, dtype = torch.float32)

mse_torch = F.mse_loss(y_pred_cont_t, y_true_cont_t).item()
chi2_torch = torch.sum((y_true_cont_t - y_pred_cont_t)**2 / (y_pred_cont_t + 1e-7)).item()
bce_torch = F.binary_cross_entropy(y_pred_bin_t, y_true_bin_t).item()
cce_torch = F.cross_entropy(y_pred_cat_t.log(), y_true_cat_t).item()  # log for torch cross_entropy expects logits



# Print comparisons ...........................................................

print("=== MSE ===")
print("Manual: ", mse_manual(y_true_cont, y_pred_cont))
print("TF    : ", mse_tf)
print("Torch : ", mse_torch)

print("\n=== Chi-squared ===")
print("Manual: ", chi2_manual(y_true_cont, y_pred_cont))
print("TF    : ", chi2_tf)
print("Torch : ", chi2_torch)

print("\n=== Binary Cross-Entropy ===")
print("Manual: ", binary_ce_manual(y_true_bin, y_pred_bin))
print("TF    : ", bce_tf)
print("Torch : ", bce_torch)

print("\n=== Categorical Cross-Entropy ===")
print("Manual: ", categorical_ce_manual(y_true_cat, y_pred_cat))
print("TF    : ", cce_tf)
print("Torch : ", cce_torch)



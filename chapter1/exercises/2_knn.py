# Introduction to Machine Learning
# Jesús Urtasun Elizari: MRC LMS 2026
# Chapter 1: What is learning



# Exercise 2: 
# Implement a knn classification
# Check calculation with tensorflow / pytorch implementation



# Import libraries ............................................................

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter




# Generate example input ......................................................

print("\nGenerate example input")

# Example data: rows represent cells, columns represent features (e.g. gene expression)
data = np.array([
    [5.1, 1.4, 0.2], # Cell type A
    [4.9, 1.4, 0.2],
    [6.2, 5.4, 2.3], # Cell type B
    [5.9, 5.1, 1.8],
    [8.2, 5.4, 5.3], # Cell type C
    [7.9, 4.1, 4.8]])
print("\nData:\n", data)
print("Shape:", data.shape)
print("Format: ", type(data))

# Labels for training: cell types
labels = np.array(['A', 'A', 'B', 'B', 'C', 'C'])
print("\nLabels: ", labels)
print("Shape:", labels.shape)
print("Format: ", type(labels))

# New cell to classify
# x_new = np.array([4.8, 1.0, 0.5])
# x_new = np.array([5.8, 5.0, 1.7])
x_new = np.array([7.8, 5.0, 5.5])
print("\nPoint to classify: ", x_new)
print("Shape:", x_new.shape)
print("Format: ", type(x_new))



# KNN implementation ..........................................................

print("\nKNN running")

def knn_predict(data, labels, x_new, k = 3):

    # Compute Euclidean distances from x_new to all cells
    # Argument axis = 1 tells numpy to compute distance row by row
    distances = np.linalg.norm(data - x_new, axis = 1)
    print("Distance to cells in data:", distances)
    
    # Sort distances from smallest to largest
    sorted_indices = distances.argsort()
    # Take the first k indices
    knn_indices = sorted_indices[:k]
    print(f"Index of the {k} nearest neighbors:", knn_indices)
    
    # Get the labels of the nearest neighbors
    knn_labels = labels[knn_indices]
    print("Labels of nearest neighbors:", knn_labels)
    
    # Return the most common label (majority vote)
    # First [0]: Takes the first tuple in the list of most common elements (e.g. ('B', 2)).
    # Second [0]: Takes the label from the tuple (ignores the count).
    most_common = Counter(knn_labels).most_common(1)[0][0]

    return most_common

# Classify the new cell
label_new = knn_predict(data, labels, x_new, k = 3)
print(f"\nNew cell classified as: {label_new}")



# Introduction to Machine Learning
# Jesús Urtasun Elizari: MRC LMS 2026
# Chapter 2: Basic mathematical intuitions



# Exercise 2: 
# Implement a series of activation functions
# Check calculation with tensorflow / pytorch implementation



# Import libraries ............................................................

import numpy as np
import matplotlib.pyplot as plt



# Simulate example data .......................................................

print("\nGenerated example input")

# Example data: gene expression vs stimulus (e.g., drug dose)
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([1.2, 2.3, 2.8, 3.5, 4.1, 5.0])
print("x: ", x)
print("Shape:", x.shape)
print("Format:", type(x))
print("y: ", y)
print("Shape:", y.shape)
print("Format:", type(y))



# Linear regression fit .......................................................

# y(x) = slope * x + intercept

# Compute means
x_mean = np.mean(x)
y_mean = np.mean(y)

# Compute slope and intercept from minimizing the Residual Sum of Squares (RSS): 
# m = sum((x - x_mean)*(y - y_mean)) / sum((x - x_mean)^2)
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean)**2)
slope = numerator / denominator

# Compute intercept using: b = y_mean - m * x_mean
intercept = y_mean - slope * x_mean
print(f"\nFit: gene_expression = {slope:.2f} * stimulus + {intercept:.2f}")



# Compute residuals ...........................................................

y_fit = slope * x + intercept
residuals = y - y_fit
print("\nResiduals:", residuals)

rss = np.sum(residuals**2) # residual sum of squares
tss = np.sum((y - np.mean(y))**2) # total sum of squares
r2 = 1 - rss/tss
print(f"Residual Sum of Squares: {rss:.2f}, R2: {r2:.2f}")

# Predict gene expression for new stimuli values
x_new = np.array([1.5, 2.5, 3.5])
y_new_pred = slope * x_new + intercept
print("Predicted gene expression for new stimuli:", y_new_pred)



# Visualization ...............................................................

# Generate figure
plt.figure(figsize = (8,5))
plt.scatter(x, y, color = 'blue', label = 'Observed data')
plt.plot(x, y_fit, color = 'red', label = 'Fitted line')
plt.scatter(x_new, y_new_pred, color = 'green', marker = 'x', label = 'Predictions')

# Show residuals as vertical lines
for xi, yi, yfi in zip(x, y, y_fit):
    plt.vlines(xi, yfi, yi, color = 'gray', linestyle = 'dashed', alpha = 0.5)

# Add labels and lagend
plt.xlabel('x: stimulus')
plt.ylabel('y(x): gene expression')
plt.title('Linear Regression: Gene expression vs stimulus')
plt.legend()
plt.show()



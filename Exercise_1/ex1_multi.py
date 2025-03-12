''' Machine Online Class - Exercise 1: Linear Regression

Instructions
------------

This file contains code that helps you get started on the
linear exercise. You will need to complete the following functions
in this exercise.

warmUpExercise.py
plotData.py
gradientDescent.py
computeCost.py
gradientDescentMulti.py
computeCostMulti.py
featureNormalize.py
normalEqn.m

For this part of the Exercise, you will need to change some 
parts of the code below for various experiments (e.g., changing 
learning rates).

Initialization
'''
from pause import pause
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


## ========================== Part 1: Feature Normalization ===================================

print('Loading data ...\n')
data = pd.read_csv('/home/yatin/Octave/AndrewNg_AIndMLCourse_Python_code/Exercise_1/ex1data2.txt', sep=",", header=None)
X = np.array(data.iloc[:, 0:2].values)
y = np.array(data.iloc[:, 2].values)
y = y.reshape(len(y), 1)
m = len(y) # Number of training examples

# Print out some data points
print('First 10 examples from the dataset: \n')
for i in range(10):
    print(f'x = {X[i, :]}' + ' , ' + f'y = {y[i]}')

#pause()

# Scale features and set them to zero mean
print('Normalizing features ...\n')
from featureNormalize import featureNormalize
X, mu, sigma = featureNormalize(X)
for i in range(10):
    print(f'x = {X[i, :]}')
# Add intercept term to X
X = np.c_[np.ones((m,1)),X]
#pause()
## ========================== Part 2: Gradient descent ===================

# % ====================== YOUR CODE HERE ======================
# % Instructions: We have provided you with the following starter
# %               code that runs gradient descent with a particular
# %               learning rate (alpha).
# %
# %               Your task is to first make sure that your functions -
# %               computeCost and gradientDescent already work with
# %               this starter code and support multiple variables.
# %
# %               After that, try running gradient descent with
# %               different values of alpha and see which one gives
# %               you the best result.
# %
# %               Finally, you should complete the code at the end
# %               to predict the price of a 1650 sq-ft, 3 br house.
# %
# % Hint: By using the 'hold on' command, you can plot multiple
# %       graphs on the same figure.
# %
# % Hint: At prediction, make sure you do the same feature normalization.

print('Running gradient descent ...\n')
# Choose some alpha value
alpha = 0.01
num_iters = 400

# Init Theta and Run Gradient Descent 
theta = np.zeros((3,1))
from gradientDescentMulti import gradientDescentMulti
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

# Plot the convergence graph
plt.plot(range(len(J_history)), J_history, '-b', linewidth=2)
plt.xlabel('Number of iteration')
plt.ylabel('Cost J')
print('Close the plot to continue')
plt.show()

# Display gradient descent's result
print('Theta computed from gradient descent: \n')
print('{} \n'.format(theta))
print('\n')

# Estimate the price of a 1650 sq-ft, 3 br house
# ========================= Your Code Here =======================
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.

house_val = np.ones((1, 3))
house_val[0, 1] = (1650 - mu[0,0]) / sigma[0,0]
house_val[0, 2] = (3 - mu[0, 1]) / sigma[0,1]
print(house_val)
price = np.dot(house_val, theta)


# =====================================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n {}\n'.format(price))

pause()

# ================================================ Part 3: Normal Equations =======================================

print('Solving with normal equations ...\n')

# % ====================== YOUR CODE HERE ======================
# % Instructions: The following code computes the closed form
# %               solution for linear regression using the normal
# %               equations. You should complete the code in
# %               normalEqn.m
# %
# %               After doing so, you should complete this code
# %               to predict the price of a 1650 sq-ft, 3 br house.
# %

# %% Load Data

data = pd.read_csv('/home/yatin/Octave/AndrewNg_AIndMLCourse_Python_code/Exercise_1/ex1data2.txt', sep=",", header=None)
X = np.array(data.iloc[:,0:2].values)
y = np.array(data.iloc[:, 2].values)
y = y.reshape(len(y), 1)

m = len(y)

# Add intercept term to X 
X = np.c_[np.ones((m,1)), X]

# Calculate the parameters from the norma equation
from normalEqn import normalEqn
theta = normalEqn(X, y)

# Display normal equation's result
print('Theta computed from the normal equations: \n')
print('\n', theta)
print('\n')

# Estimate the price of a 1650 sq-ft, 3 br house
# ======================== Your Code Here ===================================
val = np.ones((1, 3))
val[0, 1] = 1650
val[0, 2] = 3
price = val @ theta

# ===========================================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n', price)




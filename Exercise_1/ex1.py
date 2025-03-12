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

For this exercise, you will not need to change any code in this file,
or any other files other than those mentioned above.

x refers to the popultion size in 10,000s
y refers to the profit in $10,000s

Initialization
'''
from pause import pause
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ========================= Part1: Basic Funtion ===============================
# Complete warmUpExercise.py

print(f'Running warmUpExercise ... \n')
print(f'5x5 Identity Matrix: \n')
from warmUpExercise import warmUpExercise
A = warmUpExercise()
print(A)

pause()

## ========================== Part2: Plotting ===================================
print('Plotting Data ...\n')

data = pd.read_csv('/home/yatin/Octave/AndrewNg_AIndMLCourse_Python_code/Exercise_1/ex1data1.txt', sep=",", header=None)
X = data.iloc[:, 0].values
y = data.iloc[:, 1].values
m = len(y) # Number of training examples

#Plot Data
# Note: You have to complete the code in plotData.py
from plotData import plotData
plotData(X,y)

pause()

## ========================== Part3: Cost and Gradient descent ===================
X = np.array(X)
y = np.array(y)
y = y.reshape(len(y), 1)

X = np.c_[np.ones((m,1)), X] # Add a column of ones to x
theta = np.zeros((2,1)) # initialize fitting parameters

# Some gradient settings 
iterations = 1500
alpha = 0.01

print(f'\n Testing the cost function ...\n')
# compute and display initial cost
from computeCost import computeCost
J = computeCost(X, y, theta)
print(f'With theta = [0; 0]\nCost Computed = {J}\n')
print(f'Expected cost value (approx) 32.07\n')

# further testing of the cost function
J = computeCost(X, y, np.array([[-1], [2]]))
print(f'\nWith theta = [-1 ; 2]\nCost computed = {J}\n')
print(f'Expected cost value (approx) 54.24\n')

pause()

print(f'\nRunning Gradient Descent ...\n')
# run gradient descent
from gradientDescent import gradientDescent
theta = gradientDescent(X, y, theta, alpha, iterations)

# Print theta to screen
print('Theta found by gradient descent:\n')
print(theta,'\n')
print('Expected theta values (approx)\n')
print(' -3.6303\n 1.1664\n\n')

# Plot the linear fit
print('Close the plot to continue')
plt.plot(X[:,1], X@theta, '-')
plt.legend(['Training Data', 'Linear Regression'])
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.array([1, 3.5]) @ theta
print('For poppulation = 35,000, we predict a profit of {}%\n'.format(predict1[0]*10000))
predict2 = np.array([1, 7]) @ theta
print('For poppulation = 70,000, we predict a profit of {}%\n'.format(predict2[0]*10000))
pause()

# =================== Part 4: Visualizing J(theta_0, theta_1) ========================
print('Visualizing J(theta_0, theta_1) ...\n')

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix 0's
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Fill out J_vals
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
        J_vals[i, j] = computeCost(X, y, t)


# Surface Plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
# Because of the way meshgrids work in the surf command, we need to 
# transpose J_vals before calling surf, or else the axes will be flipped
ax.plot_surface(theta0_vals, theta1_vals, J_vals.T)
ax.set_xlabel(''r'$\theta_0$')
ax.set_ylabel(''r'$\theta_1$')
print('Close the plot to continue')
plt.show()

# Contour Plot
#Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
plt.contour(theta0_vals, theta1_vals, J_vals.T, np.logspace(-2, 3, 20))
plt.xlabel(''r'$\theta_0$')
plt.ylabel(''r'$\theta_1$')
plt.plot(theta[0], theta[1], 'rx', markersize=10, linewidth=2)
print('Close the plot to continue')
plt.show()

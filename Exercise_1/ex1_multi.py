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

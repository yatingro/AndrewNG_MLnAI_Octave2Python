# %% Machine Learning Online Class - Exercise 2: Logistic Regression
# %
# %  Instructions
# %  ------------
# %
# %  This file contains code that helps you get started on the second part
# %  of the exercise which covers regularization with logistic regression.
# %
# %  You will need to complete the following functions in this exericse:
# %
# %     sigmoid.m
# %     costFunction.m
# %     predict.m
# %     costFunctionReg.m
# %
# %  For this exercise, you will not need to change any code in this file,
# %  or any other files other than those mentioned above.
# %

import numpy as np
import pandas as pd
from pause import pause
import matplotlib.pyplot as plt

# Load data
# The first two columns contains the X values and the third column 
# contain the label (y).

data = pd.read_csv('/home/yatin/Octave/AndrewNg_AIndMLCourse_Python_code/Exercise_2/ex2data2.txt')
X = np.array(data.iloc[:, 0:2].values)
y = np.array(data.iloc[:, 2].values).reshape(-1,1)
from plotData import plotData
plotData(X, y)
# Put some labels and legends
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

# Specified in plot order
plt.legend(["y = 1", "y = 0"])
print('Close the plot to continue')
plt.show()

# %% =========== Part 1: Regularized Logistic Regression ============
# %  In this part, you are given a dataset with data points that are not
# %  linearly separable. However, you would still like to use logistic
# %  regression to classify the data points.
# %
# %  To do so, you introduce more features to use -- in particular, you add
# %  polynomial features to our data matrix (similar to polynomial
# %  regression).
# %

# % Add Polynomial Features

# % Note that mapFeature also adds a column of ones for us, so the intercept
# % term is handled
from mapFeature import mapFeature
X = mapFeature(X[:, 0], X[:, 1])
pause()
# Initialize fitting parameters
initial_theta = np.zeros((np.size(X, 1), 1))

# Set regularization parameter lambda to 1
lmbda = 1

# Compute and display initial cost and gradient for regularized logistic
# regression
from costFunctionReg import costFunctionReg, gradFunctionReg
cost = costFunctionReg(initial_theta, X, y, lmbda)
grad = gradFunctionReg(initial_theta, X, y, lmbda)

print('Cost at initial theta (zeros): {}\n'.format(cost))
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros) - first five values only:\n')
print('{}\n'.format(grad[0:5]))
print('Expected gradients (approx) - first five values only: \n')
print('0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')
pause()

# Compute and display cost and gradient with all-ones theta and lambda = 10
test_theta = np.ones((np.size(X, 1), 1))
cost = costFunctionReg(test_theta, X, y, 10)
grad = gradFunctionReg(test_theta, X, y, 10)

print('\nCost at test theta (with lambda = 10): {}\n'.format(cost))
print('Expected cost (approx): 3.16\n')
print('Gradient at test theta - first five values only:\n')
print('{}\n'.format(grad[0:5]))
print('Expected gradients (approx) - first five values only: \n')
print('0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n')
pause()


# %% ============= Part 2: Regularization and Accuracies =============
# %  Optional Exercise:
# %  In this part, you will get to try different values of lambda and
# %  see how regularization affects the decision coundart
# %
# %  Try the following values of lambda (0, 1, 10, 100).
# %
# %  How does the decision boundary change when you vary lambda? How does
# %  the training set accuracy vary?
# %

# initialize fitting parameters
initial_theta = np.zeros((np.size(X, 1), 1))

# Set regularization parameter lambda to 1 (you should vary this)
lmbda = 1

# Optimize  
from scipy import optimize as opt
result = opt.fmin_tnc(func=costFunctionReg, x0=initial_theta.flatten(), fprime=gradFunctionReg, args=(X, y.flatten(), lmbda))
theta = result[0]
theta = theta[:, np.newaxis]

# Plot Boundary
from plotDecisionBoundary import plotDecisionBoundary
plotDecisionBoundary(theta, X, y)
plt.title('lambda = {}'.format(lmbda))

# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend(['y=1', 'y=0', 'Decision Boundary'])
print('Close the plot to continue')
plt.show()

# Compute accuracy on our training set
from predict import predict
p = predict(theta, X)

print('\ntraining Accuracy: {}\n'.format(np.mean(np.double(p==y))*100))
print('Expected accuracy (with lmbda = 1): 83.1 (approx)\n')



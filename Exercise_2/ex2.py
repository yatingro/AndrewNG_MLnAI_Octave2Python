# %% Machine Learning Online Class - Exercise 2: Logistic Regression
# 
#   Instructions
#   ------------
#  
#   This file contains code that helps you get started on the logistic
#   regression exercise. You will need to complete the following functions 
#   in this exericse:
# 
#      sigmoid.m
#      costFunction.m
#      predict.m
#      costFunctionReg.m
# 
#   For this exercise, you will not need to change any code in this file,
#   or any other files other than those mentioned above.
# 



#%% Import liberaries
from pause import pause
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Load data
# The first two coloumns contains the exam scores and the third coloumn
# contains the label

data = pd.read_csv('/home/yatin/Octave/AndrewNg_AIndMLCourse_Python_code/Exercise_2/ex2data1.txt')
X = np.array(data.iloc[:, 0:2].values)
y = np.array(data.iloc[:, 2].values).reshape(-1,1) 

#%% ========================================== Part 1: Plottting =========================================================
# We start the exercise by first plotting the data to understand the problem 
# we are working with.

print('Plotting data with + indicatiing (y = 1) examples and o '
'indicating (y = 0) examples \n')
from plotData import plotData
plotData(X, y)

# Put some labels and legends
plt.xlabel('Exam 1')
plt.ylabel('Exam 2')

# Specified in plot order
plt.legend(["Admitted", "Not Admitted"])
print('Close the plot to continue')
plt.show()
pause()

# %% ========================================== Part 2: Compute Cost and Gradient =========================================== 
# In this part of Exercise, you will implement the cost and gradient 
# for logistic regression. You need to complete the code in 
# costFunction.m

# Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape
# Add intercept term to x and X_test
X = np.c_[np.ones((m,1)), X]



# initialize fitting parameters
initial_theta = np.zeros((n+1, 1))

# Compute and display initial cost and gradient
from costFunction import costFunction, gradient
cost = costFunction(initial_theta, X, y)
grad = gradient(initial_theta, X, y)
print('Cost at initial theta (zeros): {}\n'.format(cost))
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros): \n')
print('{}\n'.format(grad))
print('Expected gradients (approx): \n -0.1000\n -12.0092\n -11.2628\n')

# Compute and display cost and gradient with non-zero theta 
import numpy as np
test_theta = np.array([[-24], [0.2], [0.2]])
cost = costFunction(test_theta, X, y)
grad = gradient(test_theta, X, y)
print('\n Cost at test theta: {}\n'.format(cost))
print('Expected cost (approx): 0.218\n')
print('Gradient at test theta: \n')
print('{}\n'.format(grad))
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

pause()

# %% ======================== Part 3: Optimizing using fminunc ===================
# In this exercise, you will use a built-in-function (fminunc) to find the 
# optimal parameters theta.
from scipy import optimize as opt
# Optimize the theta, this function will return cost and theta
result = opt.fmin_tnc(func=costFunction, x0=initial_theta.flatten(), args=(X, y.flatten()), fprime=gradient)
theta = result[0]
theta = theta[:, np.newaxis]
cost = costFunction(theta, X, y)
# Print theta to screen 
print('Cost at theta found by fminunc: {}\n'.format(cost))
print('Expected cost (approx): 0.203\n')
print('theta: {}\n'.format(theta))
print('Expected theta (approx): \n')
print(' -25.161\n0.206\n0.201\n')

# Plot Boundary
from plotDecisionBoundary import plotDecisionBoundary
plotDecisionBoundary(theta, X, y)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
print('Close the plot to continue')
plt.show()

pause()

# %% ======================================= Part 4: Predict and Accuracies ============================
# After learning the parameters, you'll like to use it to predict the outcomes 
# on unseen data. In this part, you will use the logistic regression model 
# to predict the probability that a student with score 45 on exam 1 and 
# score 85 on exam 2 will be admitted.

# Furthermore, you will compute the training and test set accuracies of
# our model.

# Your task is to complete the code in predict.m

# Predict probability for a student with score 45 on exam 1 
# and score 85 on exam 2
from sigmoid import sigmoid
prob = sigmoid(np.array([1, 45, 85]) @ theta)
print('For a student with scores 45 and 85, we predicdt an adimission'
'probability of {}\n'.format(prob))
print('Expected Value: 0.775 +/- 0.002 \n\n')

# Compute accuracy on our training set 
from predict import predict
p = predict(theta, X)

print('Train Accuracy: {}\n'.format(np.mean(np.double(p == y)) * 100))
print('Expected accuracy (approx): 89.0\n')
print('\n')


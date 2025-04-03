import numpy as np
from sigmoid import sigmoid

def lrCostFunction(theta, X, y, lmbda):
    """ %LRCOSTFUNCTION Compute cost and gradient for logistic regression with
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. """

    # Initialize some useful values
    m = len(y)  # number of training examples

    # You need to return the following variables correctly
    J = 0
    

    h = sigmoid(X @ theta)  # hypothesis
    
    J = (-1/m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h)) + (lmbda/(2*m)) * (theta[1:].T @ theta[1:])  # cost function
    return J

def gradient(theta, X, y, lmbda):
    """ %GRADIENT Compute gradient for logistic regression with regularization"""
    m = len(y)  # number of training examples
    grad = np.zeros(theta.shape)
    # Compute the gradient
    h = sigmoid(X @ theta)  # hypothesis
    grad[0] = (1/m) * (X[:, 0].T @ (h - y))  # gradient for theta_0
    grad[1:] = (1/m) * (X[:, 1:].T @ (h - y)) + (lmbda/m) * theta[1:]  # gradient for theta_1 to theta_n
    
    return grad.flatten()

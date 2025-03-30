import numpy as np
from sigmoid import sigmoid

def costFunctionReg(theta, X, y, lmbda):
    """ %COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. """

    # Initialize the some useful values
    m = len(y) # number of training examples

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(np.shape(theta))

    z = X @ theta
    h = sigmoid(z)

#     % ====================== YOUR CODE HERE ======================
# % Instructions: Compute the cost of a particular choice of theta.
# %               You should set J to the cost.
# %               Compute the partial derivatives and set grad to the parti

    J = (1/m) * (-y.T @ np.log(h) - (1 - y).T @ np.log(1 - h)) + (lmbda/(2*m)) * np.sum(theta[1:]**2)
    return J

def gradFunctionReg(theta, X, y, lmbda):
    """ %COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. """

    # Initialize the some useful values
    m = len(y) # number of training examples

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(np.shape(theta))

    z = X @ theta
    h = sigmoid(z)
    #grad = (1/m) * X.T @ (h - y) + (lmbda/m) * np.r_['0,2', theta[0], theta[1:]]
    grad[0] = (1/m) * X[:,0].T @ (h - y)
    grad[1:] = (1/m) * X[:,1:].T @ (h - y) + (lmbda/m) * theta[1:]
    return grad

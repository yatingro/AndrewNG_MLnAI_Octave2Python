import numpy as np
def computeCostMulti(X, y, theta):
    '''Compute cost for linear regression with multiple variables 
    J = ComputeCostMULTI(X, y, theta) computes the cost of using theta as the 
    parameter for linear regression to fit the data points in X and y'''

    # initialize some useful values
    m = len(y) # number of training examples
    h = np.matmul(X, theta) # compute the prediction
    
    # You need to return the following variables correctly
    J = 1/(2*m) * np.sum((h - y)**2)
    return J
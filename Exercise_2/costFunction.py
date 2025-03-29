import numpy as np
from sigmoid import sigmoid
from pause import pause
def costFunction(theta, X, y):
    """  %COSTFUNCTION Compute cost and gradient for logistic regression
 %   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
 %   parameter for logistic regression and the gradient of the cost
 %   w.r.t. to the parameters.
 """
    # Initialize some useful values
    m = np.size(y) # number of training examples

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(np.shape(theta)) 
#% ====================== YOUR CODE HERE ======================
# % Instructions: Compute the cost of a particular choice of theta.
# %               You should set J to the cost.
# %               Compute the partial derivatives and set grad to the partial
# %               derivatives of the cost w.r.t. each parameter in theta
# %
# % Note: grad should have the same dimensions as theta
# % 

    z = X @ theta
    h = sigmoid(z)

    J = (1/m) * (-y.T @ np.log(h) - (1-y.T) @ np.log(1-h))
    grad = (1/m) * X.T @ (h-y)
    return J

def gradient(theta, X, y):
    m = np.size(y) # number of training examples
    grad = np.zeros(np.shape(theta))
    z = X @ theta
    h = sigmoid(z)
    grad = (1/m) * X.T @ (h-y)
    return grad

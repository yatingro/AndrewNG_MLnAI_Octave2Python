import numpy as np
from computeCostMulti import computeCostMulti
def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """GRADIENTDESCENTMULTI performs gradient descent to learn theta
    theta = GRADIENTDESCENTMULTI(X, y, theta, alpha, num_iters) update theta by
    taking num_iters gradient steps with learning rate alpha"""

    # Initialize some useful values
    m = len(y) # number of training examples 
    J_history = np.zeros((num_iters, 1))

    for iter in range(num_iters):
    #     % ====================== YOUR CODE HERE ======================
    # % Instructions: Perform a single gradient step on the parameter vector
    # %               theta.
    # %
    # % Hint: While debugging, it can be useful to print out the values
    # %       of the cost function (computeCostMulti) and gradient here.
    # %
        temp = np.zeros((np.size(theta, axis=0),1))
        pred = np.matmul(X, theta)
        loss = np.subtract(pred, y)
        
        temp[0] = theta[0,0] - alpha * (1/m) * np.dot(loss.T, X[:,0].reshape(m,1))
        temp[1] = theta[1,0] - alpha * (1/m) * np.dot(loss.T, X[:,1].reshape(m,1))
        temp[2] = theta[2,0] - alpha * (1/m) * np.dot(loss.T, X[:,2].reshape(m,1))

        theta[0,0] = temp[0]
        theta[1,0] = temp[1]
        theta[2,0] = temp[2]

    # =======================================================================
        # Save the cost J in every iteration
        J_history[iter] = computeCostMulti(X, y , theta)    
    
    return theta, J_history
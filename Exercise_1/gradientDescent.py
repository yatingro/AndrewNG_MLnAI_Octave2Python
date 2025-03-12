import numpy as np
from computeCost import computeCost
def gradientDescent(X, y, theta, alpha, num_iters):
    '''GradientDescent Performs gradient descent to learn theta
theta =GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
taking num_iters gradient steps with learning rate alpha. '''

    # Intialize some useful values
    m = len(y)
    J_history = np.zeros((num_iters,1)) 
    temp = np.zeros((2,1))
    
    for iter in range(1,num_iters):

        #========================YOUR CODE HERE==================
        #Instructions: Perform a single gradient step on the parameter vector theta.
        #
        #Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #
        
        pred = np.matmul(X, theta)
        loss = pred - y
        temp[0] = theta[0] - alpha * (1/m) * np.dot(loss.T,X[:,0].reshape(m,1))
        temp[1] = theta[1] - alpha * (1/m) * np.dot(loss.T,X[:,1].reshape(m,1))
        theta[0] = temp[0]
        theta[1] = temp[1]


        # =================================================================================
        # Save the cost J in every iteration
        J_history[iter] = computeCost(X, y, theta)
    
    return theta
        

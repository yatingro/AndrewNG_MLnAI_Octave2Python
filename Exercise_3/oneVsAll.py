import numpy as np
def oneVsAll(X, y, num_labels, lmbda):
    """ %ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds
%   to the classifier for label i """

    # Some usefull variables
    m, n = X.shape  # number of training examples and number of features

    from lrCostFunction import lrCostFunction, gradient
    # Add ones to the X data matrix
    X = np.column_stack((np.ones(m), X))  # add intercept term
    # Initialize all_theta
    all_theta = np.zeros((num_labels, n + 1))  # theta for each class
   
    # Set options for fminunc
    options = {'maxiter': 50}  # maximum number of iterations
    # Create a binary vector for each class
    y_matrix = np.eye(num_labels)[y.flatten()]  # one-hot encoding
    # Loop over each class
    for c in range(num_labels):
        # Set the initial theta for class c
        initial_theta = np.zeros((n + 1, 1))  # initial theta for optimization
        # Optimize the cost function
        from scipy.optimize import minimize
        res = minimize(fun=lrCostFunction, x0=initial_theta.flatten(), args=(X, y_matrix[:, c].flatten(), lmbda), method='TNC', jac=gradient)#, options=options)
        all_theta[c, :] = res.x  # store the optimized theta for class c
    return all_theta

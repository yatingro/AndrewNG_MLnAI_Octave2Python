import numpy as np
from sigmoid import sigmoid
def predictOneVsAll(all_theta, X):
    """
    %PREDICT Predict the label for a trained one-vs-all classifier. The labels
%are in the range 1..K, where K = size(all_theta, 1).
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples)"""
    m = X.shape[0]
    
    # Add intercept term to X
    X = np.hstack((np.ones((m, 1)), X))
    
    # Compute the hypothesis
    h = sigmoid(X @ all_theta.T)
    
    # Get the index of the maximum value in each row
    p = np.argmax(h, axis=1)
    
    return p + 1  # Adjust for MATLAB's 1-based indexing

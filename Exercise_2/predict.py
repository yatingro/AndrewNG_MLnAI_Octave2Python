import numpy as np
def predict(theta, X):
    """ %PREDICT Predict whether the label is 0 or 1 using learned logistic
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1) """
    m = X.shape[0] # Number of training examples
    p = np.zeros((m, 1))
    from sigmoid import sigmoid
    p = sigmoid(np.dot(X, theta)) >= 0.5
    return p

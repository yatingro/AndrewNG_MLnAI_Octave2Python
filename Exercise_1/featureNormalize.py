import numpy as np
def featureNormalize(X):
    '''FEATURENORMALIZE(X) returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation 
    is 1. This is often a good preprocessing step to do when 
    working with learning algorithms.
    '''
    # You need to set these values correctly
    X_norm = X
    mu = np.zeros((1, np.size(X, axis=1)))
    sigma = np.zeros((1, np.size(X, axis=1)))

    # ======================= YOUR CODE HERE ================================
    # Instructions: First, for each feature dimension, compute the mean 
    #               of the feature and subtract it from the dataset, storing
    #               the mean value in mu. Next, compute the standard deviation
    #               of each feature and divide each feature by it's standard 
    #               deviation, storing the standard deviation in sigma.
    # 
    #               Note that X is a matrix where each column is a feature
    #               and each row is an example. You need to perform the 
    #               normalization separately for each feature.
    # 
    # Hint: You might find the 'mean' and 'std' functions useful.
    # 

    mu[0, 0] = np.mean(X[:, 0])
    mu[0, 1] = np.mean(X[:, 1])

    sigma[0, 0] = np.std(X[:, 0])
    sigma[0, 1] = np.std(X[:, 1])
    X_norm[:, 0] = np.divide(np.subtract(X[:,0], mu[0,0]), sigma[0, 0])
    X_norm[:, 1] = np.divide(np.subtract(X[:,1], mu[0,1]), sigma[0, 1])
    return X_norm, mu, sigma

    
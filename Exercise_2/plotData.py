import matplotlib.pyplot as plt
import numpy as np
def plotData(X, y):
    """%PLOTDATA Plots the data points X and y into a new figure
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.
 """
    # Find the indices of positive and negative 
    pos = np.where(y==1)
    neg = np.where(y==0)

    plt.scatter(X[pos, 0], X[pos, 1], c='k', marker='+', linewidths=2)
    plt.scatter(X[neg, 0], X[neg, 1], c='y', marker='o', linewidths=2)
    # Put some labels and legends
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    # Specified in plot order
    plt.legend(["Admitted", "Not Admitted"])
    
    plt.show(block=False)
    plt.get_backend()
    

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def displayData(X, example_width):
    """ %DISPLAYDATA Display 2D data in a nice grid
%   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
%   stored in X in a nice grid. It returns the figure handle h and the 
%   displayed array if requested. """

# set example_width automatically if not passed in
    if not 'example_width' in locals():
        example_width = int(round(np.sqrt(X.shape[1])))
    # Gray Image
    plt.set_cmap('gray')
    m, n = X.shape
    example_height = int(n / example_width)
    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))
    # Between images padding
    pad = 1
    # Setup blank display
    display_array = -np.ones((pad + display_rows * (example_height + pad), pad + display_cols * (example_width + pad)))
    
    # Copy each example into a patch on the display array   
    curr_ex = 1
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex > m:
                break
            # Copy the patch
            # Get the max value of the patch
            max_val = np.max(np.abs(X[curr_ex - 1, :]))
            display_array[pad + j * (example_height + pad) + np.arange(example_height), pad + i * (example_width + pad) + np.arange(example_width)[:, np.newaxis]] = X[curr_ex - 1, :].reshape((example_height, example_width)) / max_val
            curr_ex += 1
        if curr_ex > m:
            break

    # Display Image
    h = plt.imshow(display_array, vmin=-1, vmax=1)
    # Do not show axis
    plt.axis('off') 
    plt.show()
    return h, display_array 


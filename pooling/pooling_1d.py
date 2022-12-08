import numpy as np

def pool_forward(A_prev, hparameters, mode = "max"):
    # Retrieve dimensions from the input shape
    (m, n_W_in) = A_prev.shape
    # Retrieve hyperparameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]
    # Define the dimensions of the output
    n_W_out = int(1 + (n_W_in - f) / stride)
    # Initialize output matrix A
    A = np.zeros((m, n_W_out))              
    for i in range(m):                         # loop over the training examples
        for w in range(n_W_out):                 # loop on the horizontal axis of the output volume
            # Find the corners of the current "slice"¡
            horiz_start = w*stride
            horiz_end = w*stride + f
            # Use the corners to define the current slice on the ith training example of A_prev, channel c.
            a_prev_slice = A_prev[i, horiz_start:horiz_end]
            # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
            if mode == "max":
                A[i, w] = np.max(a_prev_slice)
            elif mode == "average":
                A[i, w] = np.mean(a_prev_slice)    
    # Store the input and hparameters in "cache" for pool_backward()
    cache = (A_prev, hparameters)
    # Making sure your output shape is correct
    assert(A.shape == (m, n_W_out))
    return A, cache

def create_mask_from_window(x):
    mask = x == np.max(x)
    return mask

def distribute_value(dz, shape):   
    # Retrieve dimensions from shape
    (n_H, n_W) = shape
    # Compute the value to distribute on the matrix
    average = dz / (n_H * n_W)
    # Create a matrix where every entry is the "average" value
    a = np.ones(shape) * average    
    return a

def pool_backward(dA, cache, mode = "max"):
    # Retrieve information from cache
    (A_prev, hparameters) = cache
    # Retrieve hyperparameters from "hparameters"
    stride = hparameters["stride"]
    f = hparameters["f"]
    # Retrieve dimensions from A_prev's shape and dA's shape
    m, n_W_prev = A_prev.shape
    m, n_W = dA.shape
    # Initialize dA_prev with zeros
    dA_prev = np.zeros(A_prev.shape)
    for i in range(m):             # loop over the training examples
        # select training example from A_prev
        a_prev = A_prev[i]
        for w in range(n_W):               # loop on the horizontal axis
            # Find the corners of the current "slice"
            horiz_start = w
            horiz_end = horiz_start + f
            # Compute the backward propagation in both modes.
            if mode == "max":
                # Use the corners and "c" to define the current slice from a_prev
                a_prev_slice = a_prev[horiz_start:horiz_end]
                # Create the mask from a_prev_slice
                mask = create_mask_from_window(a_prev_slice)
                # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA)
                dA_prev[i, horiz_start:horiz_end] += np.multiply(mask, dA[i, w])   
            elif mode == "average":
                # Get the value a from dA
                da = dA[i, w]
                # Define the shape of the filter as fxf
                shape = (f)
                # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da.
                dA_prev[i, horiz_start:horiz_end] += distribute_value(da, shape)                
    # Making sure your output shape is correct
    assert(dA_prev.shape == A_prev.shape)
    return dA_prev
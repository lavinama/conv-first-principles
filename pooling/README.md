# Pooling
## Table of Contents
* [pooling_1d](#pooling_1d)
  * [`pool_1d_forward`](#pool_1d_forward)
  * [`pool_1d_backward`](#pool_1d_backward)
  * [`create_mask_from_window`](#create_mask_from_window)
  * [`distribute_value`](#distribute_value)
* [pooling_3d](#pooling_2d)
  * [`pool_3d_forward`](#pool_3d_forward)
  * [`pool_3d_backward`](#pool_3d_backward)

## pooling_1d

### `pool_1d_forward`

```python
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
```
[Back to top of page](#table-of-contents) <br />
[Home](https://github.com/lavinama/conv-first-principles#readme

### `pool_1d_backward`

```python
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
```
[Back to top of page](#table-of-contents) <br />
[Home](https://github.com/lavinama/conv-first-principles#readme

### `create_mask_from_window`

```python
def create_mask_from_window(x):
    mask = x == np.max(x)
    return mask
```
[Back to top of page](#table-of-contents) <br />
[Home](https://github.com/lavinama/conv-first-principles#readme

### `distribute_value`

```python
def distribute_value(dz, shape):   
    # Retrieve dimensions from shape
    (n_H, n_W) = shape
    # Compute the value to distribute on the matrix
    average = dz / (n_H * n_W)
    # Create a matrix where every entry is the "average" value
    a = np.ones(shape) * average    
    return a
```
[Back to top of page](#table-of-contents) <br />
[Home](https://github.com/lavinama/conv-first-principles#readme

## pooling_3d

### `pool_3d_forward`

```python
def pool_forward(A_prev, hparameters, mode = "max"):
    """
    Implements the forward pass of the pooling layer
    
    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
    """
    
    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve hyperparameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    
    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))              
    
    ### START CODE HERE ###
    for i in range(m):                         # loop over the training examples
        for h in range(n_H):                     # loop on the vertical axis of the output volume
            for w in range(n_W):                 # loop on the horizontal axis of the output volume
                for c in range (n_C):            # loop over the channels of the output volume
                    
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h*stride
                    vert_end = h*stride +f
                    horiz_start = w*stride
                    horiz_end = w*stride + f
                    
                    # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end,c]
                    
                    # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)
        
    # Store the input and hparameters in "cache" for pool_backward()
    cache = (A_prev, hparameters)
    
    # Making sure your output shape is correct
    assert(A.shape == (m, n_H, n_W, n_C))
    
    return A, cache
```
[Back to top of page](#table-of-contents) <br />
[Home](https://github.com/lavinama/conv-first-principles#readme

### `pool_3d_backward`

```python
def pool_backward(dA, cache, mode = "max"):
    """
    Implements the backward pass of the pooling layer
    
    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """    
    # Retrieve information from cache (≈1 line)
    (A_prev, hparameters) = cache
    
    # Retrieve hyperparameters from "hparameters" (≈2 lines)
    stride = hparameters["stride"]
    f = hparameters["f"]
    
    # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    
    # Initialize dA_prev with zeros (≈1 line)
    dA_prev = np.zeros(A_prev.shape)
    
    for i in range(m):                       # loop over the training examples
        # select training example from A_prev (≈1 line)
        a_prev = A_prev[i]
        for h in range(n_H):                   # loop on the vertical axis
            for w in range(n_W):               # loop on the horizontal axis
                for c in range(n_C):           # loop over the channels (depth)
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f
                    
                    # Compute the backward propagation in both modes.
                    if mode == "max":
                        # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        # Create the mask from a_prev_slice (≈1 line)
                        mask = create_mask_from_window(a_prev_slice)
                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])
                        
                    elif mode == "average":
                        # Get the value a from dA (≈1 line)
                        da = dA[i, h, w, c]
                        # Define the shape of the filter as fxf (≈1 line)
                        shape = (f, f)
                        # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(da, shape)    
    # Making sure your output shape is correct
    assert(dA_prev.shape == A_prev.shape)
    
    return dA_prev
```
[Back to top of page](#table-of-contents) <br />
[Home](https://github.com/lavinama/conv-first-principles#readme
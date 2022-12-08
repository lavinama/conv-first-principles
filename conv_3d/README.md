# Convolution 3D
## Table of Contents
* [conv_3d](#conv_3d)
  * [`conv_3D_forward`](#conv_3d_forward)
  * [`conv_3D_backward`](#conv_3d_backward)


## conv_3d

### `conv_3D_forward`

```python
def conv_3D_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation of the previous layer.
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_in)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_in)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    Returns:
    Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
    """
    # Element-wise product between a_slice and W. Do not add the bias yet.
    s = np.multiply(a_slice_prev,W)
    # Sum over all entries of the volume s.
    Z = np.sum(s)
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z = Z + b.astype(float)
    return Z

def conv_3D_forward(A_prev, W, b, hparameters):
    """    
    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_in, n_W_in, n_C_in)
    W -- Weights, numpy array of shape (f, f, n_C_in, n_C_out)
    b -- Biases, numpy array of shape (1, 1, 1, n_C_out)
    hparameters -- python dictionary containing "stride" and "pad"    
    
    Returns:
    Z -- conv output, numpy array of shape (m, n_H_out, n_W_out, n_C_out)
    cache -- cache of values needed for the conv_backward() function
    """
    # Retrieve dimensions from A_prev's shape
    (m, n_H_in, n_W_in, n_C_in) = A_prev.shape
    # Retrieve dimensions from W's shape
    (f, f, n_C_in, n_C_out) = W.shape
    # Retrieve information from "hparameters"
    stride = hparameters['stride']
    pad = hparameters['pad']
    # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
    n_H_out = int((n_H_in + 2*pad - f)/stride) + 1
    n_W_out =int((n_W_in + 2*pad - f)/stride) + 1
    # Initialize the output volume Z with zeros.
    Z = np.zeros([m, n_H_out, n_W_out, n_C_out])
    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev, pad)
    
    for i in range(m):                               # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i,:,:,:]                              # Select ith training example's padded activation
        for h in range(n_H_out):                           # loop over vertical axis of the output volume
            for w in range(n_W_out):                       # loop over horizontal axis of the output volume
                for c in range(n_C_out):                   # loop over channels (= #filters) of the output volume
                    # Find the corners of the current "slice"
                    vert_start = h*stride
                    vert_end = h*stride + f
                    horiz_start = w*stride 
                    horiz_end = w*stride + f
                    
                    # Use the corners to define the (3D) slice of a_prev_pad.
                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    
                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron.
                    Z[i, h, w, c] = conv_3D_single_step(a_slice_prev, W[:, :, :, c], b[:,:,:,c])

    # Making sure your output shape is correct
    assert(Z.shape == (m, n_H_out, n_W_out, n_C_out))
    
    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)
    
    return Z, cache
```
[Back to top of page](#table-of-contents) <br />
[Home](https://github.com/lavinama/conv-first-principles#readme)

### `conv_3D_backward`

```python
def conv_3D_backward(dZ, cache):
    """
    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()
    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """
    # Retrieve information from "cache"
    (A_prev, W, b, hparameters) = cache
    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape
    # Retrieve information from "hparameters"
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    # Retrieve dimensions from dZ's shape
    (m, n_H, n_W, n_C) = dZ.shape
    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                           
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))
    # Pad A_prev and dA_prev
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    for i in range(m):                       # loop over the training examples
        # select ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(n_H):                   # loop over vertical axis of the output volume
            for w in range(n_W):               # loop over horizontal axis of the output volume
                for c in range(n_C):           # loop over the channels of the output volume
                    # Find the corners of the current "slice"
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f
                    # Use the corners to define the slice from a_prev_pad
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i, h, w, c]
        # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
    # Making sure your output shape is correct
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    return dA_prev, dW, db
```
[Back to top of page](#table-of-contents) <br />
[Home](https://github.com/lavinama/conv-first-principles#readme)

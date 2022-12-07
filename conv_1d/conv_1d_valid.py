import numpy as np
from padding.np_pad import zero_pad_1d as zero_pad

def conv_1D_single_step(a_slice_prev, W, b):
    # Element-wise product between a_slice and W. Do not add the bias yet.
    s = np.multiply(a_slice_prev,W)
    # Sum over all entries of the volume s.
    Z = np.sum(s)
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z = Z + b.astype(float)
    return Z

def conv_1D_forward(A_prev, W, b=None, hparameters=None):
    # Retrieve dimensions from A_prev's shape
    (m, n_W_in) = A_prev.shape
    # Retrieve dimensions from W's shape
    (f,) = W.shape
    if not b:
        b = np.zeros([1, 1, 1], dtype=np.double)
    if not hparameters:
        hparameters = {}
        hparameters['stride'] = 1
        hparameters['pad'] = 1
    # Retrieve information from "hparameters"
    stride = hparameters['stride']
    pad = hparameters['pad']
    # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor.
    n_W_out = max(n_W_in, f) - min(n_W_in, f) + 1
    # Initialize the output volume Z with zeros.
    Z = np.zeros([m, n_W_out])
    # Create A_prev_pad by padding A_prev
    # A_prev_pad = zero_pad(A_prev, pad)
    A_prev_pad = A_prev
    for i in range(m):                               # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i,:]                              # Select ith training example's padded activation
        for w in range(n_W_out):                       # loop over horizontal axis of the output volume
            # Find the corners of the current "slice"
            horiz_start = w*stride 
            horiz_end = w*stride + f
            # Use the corners to define the (3D) slice of a_prev_pad.
            a_slice_prev = a_prev_pad[horiz_start:horiz_end]
            # Convolve the (1D) slice with the correct filter W and bias b, to get back one output neuron.
            Z[i, w] = conv_1D_single_step(a_slice_prev, W, b)
    # Making sure your output shape is correct
    assert(Z.shape == (m, n_W_out))
    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)
    return Z, cache

def conv_1D_backward(dZ, cache):
    # Retrieve information from "cache"
    (A_prev, W, b, hparameters) = cache
    # Retrieve dimensions from A_prev's shape
    (m, n_W_prev) = A_prev.shape
    # Retrieve dimensions from W's shape
    (f,) = W.shape
    # Retrieve information from "hparameters"
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    # Retrieve dimensions from dZ's shape
    (m, n_W) = dZ.shape
    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros((m, n_W_prev))                           
    dW = np.zeros((f, ))
    db = np.zeros((1, ))
    # Pad A_prev and dA_prev
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    for i in range(m):                       # loop over the training examples
        # select ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for w in range(n_W):               # loop over horizontal axis of the output volume
            # Find the corners of the current "slice"
            horiz_start = w
            horiz_end = horiz_start + f
            # Use the corners to define the slice from a_prev_pad
            a_slice = a_prev_pad[horiz_start:horiz_end, :]
            # Update gradients for the window and the filter's parameters using the code formulas given above
            da_prev_pad[horiz_start:horiz_end, :] += W[:] * dZ[i, w]
            dW[:] += a_slice * dZ[i, w]
            db[:] += dZ[i, w]
        # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
        dA_prev[i, :] = da_prev_pad[pad:-pad]
    # Making sure your output shape is correct
    assert(dA_prev.shape == (m, n_W_prev))
    return dA_prev, dW, db


def conv_1d_valid(input, filter):
    n_input = input.shape[0]
    n_filter = filter.shape[0]
    n_out = n_input - n_filter + 1

    rev_filter = filter[::-1].copy()
    output = np.zeros(n_out, dtype=np.double)
    for i in range(n_out):
        output[i] = np.dot(input[i: i + n_filter], rev_filter)
    return output

if __name__=="__main__":
    np.random.seed(1)
    input = np.random.randint(0, 5, size=(20, 5)) # (N, n_W)
    filter = np.array([1, 1, 1, 3])
    print(np.convolve(input[0], filter, mode='valid'))
    print(conv_1d_valid(input[0], filter))
    filter = filter[::-1].copy()
    output, cache = conv_1D_forward(input, filter)
    print(output[0])
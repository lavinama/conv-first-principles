import tensorflow as tf
import numpy as np
from padding.pad_3d import pad_3d_no_nppad

def grader():
    ones_2d = np.ones((5,5))
    weight_2d = np.ones((3,3))
    biases_2d = np.ones((1,1))
    hparameters = {"pad" : [[0, 0], [1, 1], [1, 1], [0, 0]],
                "stride": [1, 1]}

    in_2d = tf.constant(ones_2d, dtype=tf.float32)
    filter_2d = tf.constant(weight_2d, dtype=tf.float32)
    bias_2d = tf.constant(biases_2d, dtype=tf.float32)

    in_width = int(in_2d.shape[0])
    in_height = int(in_2d.shape[1])

    filter_width = int(filter_2d.shape[0])
    filter_height = int(filter_2d.shape[1])

    input_2d = tf.reshape(in_2d, [1, in_height, in_width, 1])
    kernel_2d = tf.reshape(filter_2d, [filter_height, filter_width, 1, 1])

    conv_output_2d = tf.squeeze(tf.nn.conv2d(input_2d, kernel_2d, 
                                        strides=hparameters["stride"],
                                        padding=hparameters["pad"]))
    output_2d_gt = conv_output_2d + bias_2d
    return output_2d_gt

def conv_2D_single_step(a_slice_prev, W, b):
    # Element-wise product between a_slice and W.
    s = np.multiply(a_slice_prev,W)
    # Sum over all entries of the volume s.
    Z = np.sum(s)
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z = Z + b.astype(float)
    return Z

def conv_2D(A_prev, W, b=None, hparameters=None):
    (m, in_height, in_width) = A_prev.shape
    (filter_height, filter_width) = W.shape
    if not b:
        b = np.ones([1, 1, 1], dtype=np.double)
    if not hparameters:
        hparameters = {}
        hparameters['stride'] = 1
        hparameters['pad'] = 1
    stride = hparameters['stride']
    pad = hparameters['pad']
    
    n_H_out = int((in_height + 2*pad - filter_height)/stride) + 1
    n_W_out =int((in_width + 2*pad - filter_width)/stride) + 1
    
    Z = np.zeros([m, n_H_out, n_W_out])
    A_prev_pad = pad_3d_no_nppad(A_prev, pad)
    
    for i in range(m):                               # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i,:,:]                              # Select ith training example's padded activation
        for h in range(n_H_out):                           # loop over vertical axis of the output volume
            for w in range(n_W_out):                       # loop over horizontal axis of the output volume                    
                # Find the corners of the current "slice"
                vert_start = h*stride
                vert_end = h*stride + filter_height
                horiz_start = w*stride 
                horiz_end = w*stride + filter_width
                # Use the corners to define the (3D) slice of a_prev_pad.
                a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end]
                # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron.
                Z[i, h, w] = conv_2D_single_step(a_slice_prev, W[:, :], b[:, :])                
    # Making sure your output shape is correct
    assert(Z.shape == (m, n_H_out, n_W_out))
    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)
    return Z, cache

if __name__=="__main__":
    ones_2d = np.ones((5,5))
    weight_2d = np.ones((3,3))
    hparameters = {"pad" : 1, # [[0, 0], [1, 1], [1, 1], [0, 0]],
                "stride": 1} # [1, 1]}

    in_2d = np.array(ones_2d, dtype=np.double)
    filter_2d = np.array(weight_2d, dtype=np.double)
    in_width = int(in_2d.shape[0])
    in_height = int(in_2d.shape[1])
    filter_width = int(filter_2d.shape[0])
    filter_height = int(filter_2d.shape[1])

    input_2d = np.reshape(in_2d, [1, in_height, in_width])
    kernel_2d = np.reshape(filter_2d, [filter_height, filter_width])

    output_2d, cache = conv_2D(input_2d, kernel_2d, hparameters=hparameters)
    output_2d_gt = grader()
    print("output_2d:",  output_2d)
    print("output_2d_gt:", output_2d_gt)
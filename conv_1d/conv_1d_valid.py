import numpy as np
from padding.np_pad import zero_pad_1d as zero_pad
from padding.np_pad import *

def conv_1d_valid(input, filter):
    n_input = input.shape[0]
    n_filter = filter.shape[0]
    n_out = n_input - n_filter + 1

    output = np.zeros(n_out, dtype=np.double)
    for i in range(n_out):
        output[i] = np.dot(input[i: i + n_filter], filter)
    return output

if __name__=="__main__":
    np.random.seed(1)
    input = np.random.randint(0, 5, size=(20, 5)) # (N, n_W)
    filter = np.array([1, 1, 1, 3])
    print(np.convolve(input[0], filter[::-1], mode='valid'))
    print(conv_1d_valid(input[0], filter))
    output, cache = conv_1D_forward(input, filter)
    print(output[0])
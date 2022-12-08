from conv_1d_valid import conv_1d_valid, conv_1D_forward
import numpy as np

def conv_1d_valid_batch(input, filter, batch_size=None):
    (N, n_W_in) = input.shape
    if not batch_size:
        batch_size = N
    (f, ) = filter.shape
    n_W_out = max(n_W_in, f) - min(n_W_in, f) + 1
    output = []
    for i in range(N//batch_size):
        input_batch = input[batch_size*i : min(batch_size*(i+1), N), :]
        output.append(Z)
        Z = np.zeros([batch_size, n_W_out], dtype=np.double)
        for i in range(batch_size): # loop over the batch of training examples
            Z[i], cache = conv_1d_valid(input_batch[i,:], filter)
        output.append(Z)
    return np.array(output)

def conv_1D_batch(input, filter, batch_size=None): # input: (n, W)
    (N, n_W_in) = input.shape
    if not batch_size:
        batch_size = N
    (f, ) = filter.shape
    output = []
    for i in range(N//batch_size):
        input_batch = input[batch_size*i : min(batch_size*(i+1), N), :]
        Z, cache = conv_1D_forward(input_batch, filter)
        output.append(Z)
    return np.array(output)

if __name__=="__main__":
    np.random.seed(1)
    input = np.random.randint(0, 5, size=(20, 5)) # (N, n_W)
    filter = np.array([1, 1, 1, 3])
    output = conv_1D_batch(input, filter, batch_size=4)
    print(output) # output: (N//batch_size, batch_size, n_W_out)
from conv_1d_valid import conv_1d_valid
import numpy as np

def conv_1d_valid_batch(input, filter):
    """
    input (m, W_in)
    filter (k, )
    """
    (m, W_in) = input.shape
    (k, ) = filter.shape
    n_W_out = W_in - k + 1

    Z = np.zeros([m, n_W_out], dtype=np.double)
    for i in range(m): # loop over the batch of training examples
        single_input = input[i,:]
        Z[i] = conv_1d_valid(single_input, filter)
    return Z

if __name__=="__main__":
    np.random.seed(1)
    input = np.random.randint(0, 5, size=(20, 5)) # (N, n_W)
    filter = np.array([1, 1, 1, 3])
    batch_size = 4
    N, n_W = input.shape
    output = [] # (N//batch_size, batch_size, n_W_out)
    for i in range(N//batch_size):
        input_batch = input[batch_size*i : min(batch_size*(i+1), N), :]
        output.append(conv_1d_valid_batch(input_batch, filter))
    output = np.array(output)
    print(output) # output: (N//batch_size, batch_size, n_W_out)
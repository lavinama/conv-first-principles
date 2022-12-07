import numpy as np

def convolve_1d_full(input, filter):
    n_input = input.shape[0]
    n_filter = filter.shape[0]
    n_out = n_input + n_filter - 1

    rev_filter = filter[::-1].copy()
    output = np.zeros(n_out, dtype=np.double)
    j = 0
    for i in range(1-n_filter, n_input):
        I = input[max(0,i):min(i + n_filter, n_input)]
        K = rev_filter[max(-i,0):n_input - i*(n_input - n_filter<i)]
        output[j] = np.dot(I, K)
        j += 1
    return output

if __name__=="__main__":
    input = np.array([1, 1, 2, 2, 1])
    filter = np.array([1, 1, 1, 3])
    print(np.convolve(input, filter, mode='full'))
    print(convolve_1d_full(input, filter))
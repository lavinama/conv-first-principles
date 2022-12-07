import numpy as np

def pad_4d_no_nppad(array, padding): # array (m, h, w, c)
    # Calculate the new shape
    new_shape = np.array(array.shape) + np.array([0, int(2*padding), int(2*padding), 0])
    result = np.zeros(new_shape)
    offsets = np.array([0, padding, padding, 0])
    # Create a list of slices from offset to offset + shape in each dimension
    # Use tuple because numpy prefers using tuple for indices
    insertHere = tuple([slice(offsets[dim], offsets[dim]+array.shape[dim]) for dim in range(array.ndim)])
    # Insert the array in the result at the specified offsets
    result[insertHere] = array
    return result


if __name__=="__main__":
    # 4 Dimensions
    a = np.ones((4,3,3,3)) # (m, in_h, in_w, in_channels)
    padding = 1
    print(np.version.version)

    output = pad_4d_no_nppad(a, padding)
    print(output.shape)
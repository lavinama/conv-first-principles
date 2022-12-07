import numpy as np

def pad_3d_no_nppad(array, padding): # array: (m, h, w)
    # Calculate the new shape
    new_shape = np.array(array.shape) + np.array([0, 2*padding, 2*padding]) # (m, h, w)
    result = np.zeros(new_shape)
    offsets = np.array([0,padding, padding])
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = tuple([slice(offsets[dim], offsets[dim]+array.shape[dim]) for dim in range(array.ndim)])
    # Insert the array in the result at the specified offsets
    result[insertHere] = array
    return result

if __name__=="__main__":
    # 3 Dimensions
    a = np.ones((4,3,3))
    padding = 1
    output = pad_3d_no_nppad(a, padding)
    print(output)
# Padding
## Table of Contents
* [NpPad](#nppad)
  * [`zero_pad_1d`](#zero_pad_1d)
  * [`zero_pad_3d`](#zero_pad_3d)
  * [`zero_pad_4d`](#zero_pad_4d)
* [No NpPad](#no-nppad)
  * [`pad_3d_no_nppad`](#pad_3d_no_nppad)
  * [`pad_4d_no_nppad`](#pad_4d_no_nppad)


[Home](https://github.com/lavinama/conv-first-principles#readme)

## NpPad

### `zero_pad_1d`

```python
def zero_pad_1d(X, pad): # X: (m, W)
		# pad_with = ((before_m, after_m), (before_W, after_W))
    X_pad = np.pad(X, ((0,0), (pad,pad)), 'constant', constant_values = (0,0))
    return X_pad
```
[Back to top of page](#table-of-contents)
[Home](https://github.com/lavinama/conv-first-principles#readme)

### `zero_pad_3d`

```python
def zero_pad_3d(X, pad): # X: (m, H, W)
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
    as illustrated in Figure 1.
    
    Argument:
    X -- python numpy array of shape (m, H, W) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad)
    """
    # np.pad(array, pad_width, mode='constant', constant_values)
		# pad_with = ((before_m, after_m), (before_H, after_H), (before_W, after_W))
    X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad)), 'constant', constant_values = (0,0))
    
    return X_pad
```
[Back to top of page](#table-of-contents)
[Home](https://github.com/lavinama/conv-first-principles#readme)

### `zero_pad_4d`

```python
def zero_pad_4d(X, pad): # X: (m, H, W, C)
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
    as illustrated in Figure 1.
    
    Argument:
    X -- python numpy array of shape (m, H, W, C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """
    # np.pad(array, pad_width, mode='constant', constant_values)
		# pad_with = ((before_m, after_m), (before_H, after_H), (before_W, after_W), (before_C, after_C))
    X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), 'constant', constant_values = (0,0))
    
    return X_pad
```
[Back to top of page](#table-of-contents)
[Home](https://github.com/lavinama/conv-first-principles#readme)

## No NpPad

### `pad_3d_no_nppad`

```python
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
```
[Back to top of page](#table-of-contents)
[Home](https://github.com/lavinama/conv-first-principles#readme)

### `pad_4d_no_nppad`

```python
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
```
[Back to top of page](#table-of-contents)
[Home](https://github.com/lavinama/conv-first-principles#readme)
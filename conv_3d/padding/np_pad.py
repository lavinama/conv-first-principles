import numpy as np

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
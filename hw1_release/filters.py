"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""
import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    for row_i in range(Hi):
        for col_i in range(Wi):
            for row_k in range(Hk):
                for col_k in range(Wk):
                    roi_row = row_i + Hk//2 - row_k
                    roi_col = col_i + Wk//2 - col_k
                    if 0 <= roi_row < Hi and 0 <= roi_col < Wi:
                        out[row_i, col_i] += image[roi_row, roi_col] * kernel[row_k, col_k]
    ### END YOUR CODE
    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.zeros((H + 2 * pad_height, W + 2 * pad_width), dtype = image.dtype)
    out[pad_height: H + pad_height, pad_width: W + pad_width] = image
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    kernel = np.flip(kernel, axis=0)
    kernel = np.flip(kernel, axis=1)
    padded_image = zero_pad(image, Hk // 2, Wk // 2)
    for row_i in range(Hi):
        for col_i in range(Wi):
            out[row_i, col_i] = np.sum(kernel * padded_image[row_i: row_i + Hk,
                                                             col_i: col_i + Wk])
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    kernel = np.flip(kernel, axis=0)
    kernel = np.flip(kernel, axis=1)
    padded_image = zero_pad(image, Hk // 2, Wk // 2)
    
    for row_i in range(Hi):
        sum_cols = np.sum(kernel * padded_image[row_i: row_i + Hk, : Wk], axis=0)
        out[row_i, 0] = np.sum(sum_cols)
        
        # loop from the second column
        for col_i in range(1, Wi):
            # next column
            next_col = np.sum(kernel[:, -1] * padded_image[row_i: row_i + Hk, col_i + Wk - 1])
            out[row_i, col_i] = out[row_i, col_i - 1] - sum_cols[0] + next_col
            
            # update sum_cols
            sum_cols[:-1] = sum_cols[1:]
            sum_cols[-1] = next_col
    ### END YOUR CODE
    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g = np.flip(g, axis=0)
    g = np.flip(g, axis=1)
    out = conv_fast(f, g)
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    # calculate the mean of kernel g
    mean_g = np.mean(g)
    
    # subtract kernel g to its mean to move its mean to zero
    g -= mean_g
    out = cross_correlation(f, g)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    Hg, Wg = g.shape[:2]
    Hf, Wf = f.shape[:2]
    out = np.zeros_like(f)
    
    # zero mean the kernel g
    mean_g = np.mean(g)
    std_g = np.std(g)
    g = (g - mean_g)/std_g
    
    padded_image = zero_pad(f, Hg // 2, Wg // 2)
    for row_f in range(Hf):
        for col_f in range(Wf):
            roi = padded_image[row_f: row_f + Hg, col_f: col_f + Wg]
            mean_roi = np.mean(roi)
            std_roi = np.std(roi)
            roi = (roi - mean_roi)/std_roi
            out[row_f, col_f] = np.sum(g * roi)
    ### END YOUR CODE
    return out

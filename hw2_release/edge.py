"""
CS131 - Computer Vision: Foundations and Applications
Assignment 2
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/18/2017
Python Version: 3.5+
"""
import numpy as np
from collections import deque


def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    ### YOUR CODE HERE
    kernel = np.flip(kernel, axis=0)
    kernel = np.flip(kernel, axis=1)
    for row_i in range(Hi):
        for col_i in range(Wi):
            out[row_i, col_i] = np.sum(kernel * padded[row_i: row_i + Hk, col_i: col_i + Wk])
    ### END YOUR CODE

    return out


def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """

    kernel = np.zeros((size, size))

    ### YOUR CODE HERE
    k = size // 2
    for i in range(size):
        for j in range(size):
            kernel[i,j] = np.exp(-((i - k)**2 + (j - k)**2)/(2 * sigma**2))/(2 * np.pi * sigma**2)
    ### END YOUR CODE

    return kernel


def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[0, 0, 0],
                       [.5, 0, -.5],
                       [0, 0, 0],
                     ])
    out = conv(img, kernel)
    ### END YOUR CODE

    return out


def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[0, .5, 0],
                       [0, 0, 0],
                       [0, -.5, 0],
                     ])
    out = conv(img, kernel)
    ### END YOUR CODE

    return out


def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    grad_x = partial_x(img)
    grad_y = partial_y(img)
    G = np.sqrt(np.power(grad_x, 2) + np.power(grad_y, 2))
    theta = np.rad2deg(np.arctan2(grad_y, grad_x))
    
    # convert angle range from (-180, 180] to [0, 360]
    theta %= 360.0
    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45

    ### BEGIN YOUR CODE
    for row in range(H):
        for col in range(W):

            """
            The north and south direction are reversed because the y_axis in
            Euclidean coordinate is reversing of y_axis of image coordinate 
            nn = G[row + 1, col]
            ss = G[row - 1, col]
            ee = G[row, col + 1]
            ww = G[row, col - 1]
            ne = G[row + 1, col + 1]
            nw = G[row + 1, col - 1]
            se = G[row - 1, col + 1]
            sw = G[row - 1, col - 1]
            """
            neighbors = []
            for i in (row - 1, row, row + 1):
                for j in (col - 1, col, col + 1):
                    # print('row: {}, col: {}'.format(i, j))
                    if i < 0 or j < 0 or i >= H or j >= W:
                        # print('append 0')
                        neighbors.append(0)

                    elif i == row and j == col:
                        # print('skip')
                        continue

                    else:
                        # print('valid value')
                        neighbors.append(G[i, j])
            # print(neighbors)
            sw, ss, se, ww, ee, nw, nn, ne = neighbors

            # 0 degree
            if theta[row, col] == 0 or theta[row, col] == 180 or theta[row, col] == 360:
                if (G[row, col] >= ee) and (G[row, col] >= ww):
                    out[row, col] = G[row, col]
            
            # 45 degree
            elif theta[row, col] == 45 or theta[row, col] == 225:
                if (G[row, col] >= ne) and (G[row, col] >= sw):
                    out[row, col] = G[row, col]
            
            # 90 degree
            elif theta[row, col] == 90 or theta[row, col] == 270:
                if (G[row, col] >= nn) and (G[row, col] >= ss):
                    out[row, col] = G[row, col]
            
            # 135 degree
            elif theta[row, col] == 135 or theta[row, col] == 315:
                if (G[row, col] >= nw) and (G[row, col] >= se):
                    out[row, col] = G[row, col]
                
    ### END YOUR CODE

    return out


def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype=np.bool)
    weak_edges = np.zeros(img.shape, dtype=np.bool)

    ### YOUR CODE HERE
    H, W = img.shape
    for row in range(H):
        for col in range(W):
            if img[row, col] > high:
                strong_edges[row, col] = True

            elif img[row, col] > low:
                weak_edges[row, col] = True
    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors


def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=np.bool)

    # Make new instances of arguments to leave the original
    # references intact
    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)

    ### YOUR CODE HERE
    # loop through every strong edges
    for y_s, x_s in indices:
        # find weak edges that connected to strong edges
        neighbors_s = get_neighbors(y_s, x_s, H, W)
        for y_w, x_w in neighbors_s:
            # found weak edge
            if weak_edges[y_w, x_w]:
                edges[y_w, x_w] = True

                # Using BFS to find remaining pixel of the weak edge
                visited = {(y_w, x_w)}
                frontiers = deque([(y_w, x_w)])
                while frontiers:
                    node = frontiers.popleft()
                    neighbors_w = get_neighbors(*node, H, W)
                    for node_w in neighbors_w:
                        if node_w not in visited:
                            visited.add(node_w)
                            y_n, x_n = node_w
                            if weak_edges[y_n, x_n]:
                                edges[y_n, x_n] = True
                                frontiers.append(node_w)
    ### END YOUR CODE
    return edges


def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    ### YOUR CODE HERE
    # Gaussian blurring to reduce noise
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed = conv(img, kernel)

    # Find gradient
    G, theta = gradient(smoothed)

    # Non-maximum suppression to thin to edge
    thinned_edge = non_maximum_suppression(G, theta)

    # double thresholding
    strong_edges, weak_edges = double_thresholding(thinned_edge, high, low)

    # link weak edges to strong edges
    edge = link_edges(strong_edges, weak_edges)
    ### END YOUR CODE

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W).
        
    Returns:
        accumulator: numpy array of shape (m, n).
        rhos: numpy array of shape (m, ).
        thetas: numpy array of shape (n, ).
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    ### YOUR CODE HERE

    # loop through all edge points
    for y, x in zip(ys, xs):
        for theta_idx in range(num_thetas):
            rho = y * sin_t[theta_idx] + x * cos_t[theta_idx]
            # rounding rho
            rho = int(np.floor(rho + 0.5))

            # get rho idx in rhos
            rho_idx = rho - diag_len

            # increment accumulator
            accumulator[rho_idx, theta_idx] += 1
    ### END YOUR CODE

    return accumulator, rhos, thetas

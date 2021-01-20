import numpy as np
from itertools import product

def maxp_Forward(X, stride, kernel_size):
    N,C,H,W = X.shape
    KH, KW = kernel_size
    out_shape = (N, C, 1 + int((H - KH)/stride), 1 + int((W - KW)/stride))

    Y = np.zeros(out_shape)

    for i in range(N):                         # loop over the training examples
        for h in range(out_shape[2]):                     # loop on the vertical axis of the output volume
            for w in range(out_shape[3]):                 # loop on the horizontal axis of the output volume
                for c in range (C):            # loop over the channels of the output volume
                    
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h*stride
                    vert_end = h*stride + KH
                    horiz_start = w*stride
                    horiz_end = w*stride + KW
                    
                    # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                    X_slice = X[i, vert_start:vert_end, horiz_start:horiz_end,c]
                    
                    # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
                    Y[i, c, h, w] = np.max(X_slice)
    
    return Y

A_prev = np.random.randn(2, 3, 4, 4)
#print(A_prev,"\n")
Y=maxp_Forward(A_prev, 2, (3,3))
#print(Y)
def create_mask_from_window(x):
    mask = x == np.max(x)
    
    return mask

def distribute_value(dz, shape):
    # Retrieve dimensions from shape (≈1 line)
    (n_H, n_W) = shape
    
    # Compute the value to distribute on the matrix (≈1 line)
    average = dz / (n_H * n_W)
    
    # Create a matrix where every entry is the "average" value (≈1 line)
    a = np.ones(shape) * average
    ### END CODE HERE ###
    
    return a

arr= np.array([[[[5,4],[3,2],[1,2]]]])
mask= create_mask_from_window(arr)
#print(mask)
#print(arr.shape)
#print(np.repeat(arr, repeats=2, axis=2))
dY = np.repeat(np.repeat(arr, repeats=2, axis=2),repeats=2, axis=3)
#print(dY)

p = (5>=10)
#print(p)


def min_forward(X, kernel_size, out_channels, stride):
    N, C, H, W = X.shape
    KH, KW = kernel_size
    grad = np.zeros_like(X)
    out_shape = (N, out_channels, 1 + int((H - KH)/stride), 1 + int((W - KW)/stride))
    Y = np.zeros(out_shape)
    P = np.zeros_like(X)
    # for n in range(N):
    for h, w in product(range(0, out_shape[2]), range(0, out_shape[3])):
        h_offset, w_offset = h*stride, w*stride
        rec_field = X[:, :, h_offset:h_offset+KH, w_offset:w_offset+KW]
        # print(P[:, :, h_offset:h_offset+KH, w_offset:w_offset+KW].shape)
        Y[:, :, h, w] = np.mean(rec_field, axis=(2, 3)) 
        P[:, :, h_offset:h_offset+KH, w_offset:w_offset+KW] = (np.ones_like(rec_field) * np.mean(rec_field, axis=(2, 3))[0][0])
        #  = 
        for kh, kw in product(range(KH), range(KW)):
            #this contains the mask that will be multiplied to the max value error
            grad[:, :, h_offset+kh, w_offset+kw] = P[:, :, h_offset+kh, w_offset+kw] 

    # storing the gradient
    print(P)
    print(grad)
    print("Y: ", Y)
    return Y  
# arr= np.array([[[[31,15,28,124],[0,100,70,38],[12,12,7,2],[12,12,45,6]]]])
# forward(arr, (2,2),1, 2)

dY = np.array([[[[7, 8],[-1, 2]]]])

def min_backward(dY):
    KH, KW = (2, 2)
    N, C, H, W = dY.shape
    grad = np.zeros_like([[[[31,15,28,124],[0,100,70,38],[12,12,7,2],[12,12,45,6]]]])

    for h, w in product(range(0, H), range(0, W)):
        h_offset, w_offset = h*2, w*2
        rec_field = grad[:, :, h_offset:h_offset+KH, w_offset:w_offset+KW]
        grad[:, :, h_offset:h_offset+KH, w_offset:w_offset+KW] = (np.ones_like(rec_field) * dY[:, :, h, w].reshape(N, C, 1, 1))

    # for n in range(N):
    #     for f in range(C):
    #         for h in range(H):
    #             for w in range(W):
    #                 h_offset, w_offset = h*2, w*2
    #                 rec_field = grad[n, f, h_offset:h_offset+KH, w_offset:w_offset+KW]
    #                 grad[n, f, h_offset:h_offset+KH, w_offset:w_offset+KW] = (np.ones_like(rec_field) * dY[n, f, h, w].reshape(N, C, 1, 1))

    return (grad/(2*2))

print(min_backward(dY))

input_arr = np.asarray([[[[1, 1, 2, 4], [5, 6, 7, 8], [3, 2, 1, 0], [1, 2, 3, 4]]]])

def max_forward(X):
    N, C, H, W = X.shape
    KH, KW = (2, 2)
    grad = np.zeros_like(X)
    out_shape = (N, C, 1 + int((H - KH) / 2), 1 + int((W - KW)/2))
    Y = np.zeros(out_shape)
    # for n in range(N):
    for h, w in product(range(0, out_shape[2]), range(0, out_shape[3])):
        h_offset, w_offset = h*2, w*2
        rec_field = X[:, :, h_offset:h_offset+KH, w_offset:w_offset+KW]
        Y[:, :, h, w] = np.max(rec_field, axis=(2, 3))
        grad[:, :, h_offset:h_offset+KH, w_offset:w_offset+KW] = (X[:, :, h_offset:h_offset+KH, w_offset:w_offset+KW] >= np.max(rec_field, axis=(2, 3))[0][0])

    print(Y)
    # storing the gradient
    return grad


def max_backward(X, dY):
    KH, KW = (2, 2)
    N, C, H, W = (1, 1, 2, 2)
    for h, w in product(range(0, H), range(0, W)):
        h_offset, w_offset = h * 2, w * 2
        rec_field = X[:, :, h_offset:h_offset+KH, w_offset:w_offset+KW]
        X[:, :, h_offset:h_offset+KH, w_offset:w_offset+KW] = (rec_field * dY[:, :, h, w].reshape(N, C, 1, 1))
    print(X)


max_backward(max_forward(input_arr), np.asarray([[[[6, 8], [3, 4]]]]))

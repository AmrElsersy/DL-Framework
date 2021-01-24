from abstract_classes import *
import numpy as np
from math import sqrt, isnan
from itertools import product

###calculate_local_grads in conv


def zero_pad(X, padding_width, dims):
    """
    Pads the given array X with zeroes at the both end of given dims.
    Args:
        X: numpy.ndarray.
        padding_width: int, width of the padding.
        dims: int or tuple, dimensions to be padded.
    Returns:
        X_padded: numpy.ndarray, zero padded X.
    """
    dims = (dims) if isinstance(dims, int) else dims
    pad = [(0, 0) if idx not in dims else (padding_width, padding_width)
           for idx in range(len(X.shape))]
    X_padded = np.pad(X, pad, 'constant')
    return X_padded

class conv(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels # No. of filters to be learned
        self.stride = stride
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) \
                           else (kernel_size, kernel_size)
        self.padding = padding
        self._init_weights(in_channels, out_channels, self.kernel_size)

    def _init_weights(self, in_channels, out_channels, kernel_size):
        scale = 2/sqrt(in_channels*kernel_size[0]*kernel_size[1])

        self.weight = {'W': np.random.normal(scale=scale,
                                             size=(out_channels, in_channels, *kernel_size)),
                       'b': np.zeros(shape=(out_channels, 1))}
    
    def forward(self, X):
        """
        Forward pass for the convolution layer.
        Args:
            X: numpy.ndarray of shape (N, C, H_in, W_in) => N-> !!!!!######image no#####!!!!!, C-> No. of channels, .
        Returns:
            Y: numpy.ndarray of shape (N, F, H_out, W_out).
        """
        if self.padding:
            X = zero_pad(X, padding_width=self.padding, dims=(2, 3))

        self.cache['X'] = X

        N, C, H, W = X.shape
        
        # To get the kernel dimension
        KH, KW = self.kernel_size 
        
        #int((in_dim - f_s)/stride)+1
        out_shape = (N, self.out_channels, 1 + int((H - KH)/self.stride), 1 + int((W - KW)/self.stride))
        
        # Getting th output to have the same shape as the input
        Y = np.zeros(out_shape)
        
        for n in range(N):
            for c_w in range(self.out_channels):
                for h, w in product(range(out_shape[2]), range(out_shape[3])):
                    h_offset, w_offset = h*self.stride, w*self.stride
                    
                    rec_field = X[n, :, h_offset:h_offset + KH, w_offset:w_offset + KW]
                    # print(rec_field.shape)
                    # print((self.weight['W'][c_w]).shape)
                    Y[n, c_w, h, w] = np.sum(self.weight['W'][c_w]*rec_field) + self.weight['b'][c_w]
        assert(not isnan(np.max(Y)))

        return Y

    def backward(self, dY):
        # calculating the global gradient to be propagated backwards
        # TODO: this is actually transpose convolution, move this to a util function
        # print(dY.shape)
        # print("\nConv backward started")
        X = self.cache['X']
        dX = np.zeros_like(X)
        dX = np.asarray(dX, dtype='float64')
        N, C, H, W = dX.shape
        KH, KW = self.kernel_size
        for n in range(N): 
            for c_w in range(self.out_channels):
                for h, w in product(range(dY.shape[2]), range(dY.shape[3])):
                    h_offset, w_offset = h * self.stride, w * self.stride
                    # print("line 91",dY[n, c_w, h, w])
                    dX[n, :, h_offset:h_offset + KH, w_offset:w_offset + KW] += self.weight['W'][c_w] * dY[n, c_w, h, w]

        # calculating the global gradient wrt the conv filter weights
        dW = np.zeros_like(self.weight['W'])
        for c_w in range(self.out_channels):
            for c_i in range(self.in_channels):
                for h, w in product(range(KH), range(KW)):
                    X_rec_field = X[:, c_i, h:H-KH+h+1:self.stride, w:W-KW+w+1:self.stride]
                    dY_rec_field = dY[:, c_w]
                    dW[c_w, c_i, h, w] = np.sum(X_rec_field*dY_rec_field)

        # calculating the global gradient wrt to the bias
        db = np.sum(dY, axis=(1, 2, 3)).reshape(-1, 1)

        # caching the global gradients of the parameters
        self.weights_global_grads = {'w': dW, 'b': db}
        # print("Conv backward Ended\n")
        assert(not isnan(np.max(dX)))
        if self.padding==0:
            return dX
        else:
            return dX[:, :, self.padding:-self.padding, self.padding:-self.padding]

class AvgPool2D(Function):
    def __init__(self, kernel_size=(2, 2), stride = 2):
        super().__init__()
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.input_shape = 0
        self.grad = 0
    
    def forward(self, X):
        self.input_shape = X.shape
        N, C, H, W = X.shape
        KH, KW = self.kernel_size

        self.grad = np.zeros_like(X)
        out_shape = (N, C, 1 + int((H - KH)/self.stride), 1 + int((W - KW)/self.stride))
        Y = np.zeros(out_shape)
        #P = np.zeros_like(X)
        # for n in range(N):
        for h, w in product(range(0, out_shape[2]), range(0, out_shape[3])):
            h_offset, w_offset = h*self.stride, w*self.stride
            rec_field = X[:, :, h_offset:h_offset+KH, w_offset:w_offset+KW]
            Y[:, :, h, w] = np.mean(rec_field, axis=(2, 3))
            self.grad[:, :, h_offset:h_offset+KH, w_offset:w_offset+KW] = (np.ones_like(rec_field) * np.mean(rec_field, axis=(2, 3))[0][0])
            # for kh, kw in product(range(KH), range(KW)):
            #     #this contains the mask that will be multiplied to the max value error
            #     grad[:, :, h_offset+kh, w_offset+kw] = P[:, :, h_offset+kh, w_offset+kw] 
        # storing the gradient
        assert(not isnan(np.max(Y)))
        return Y  

    def backward(self, dY):
        KH, KW = self.kernel_size
        N, C, H, W = dY.shape

        for h, w in product(range(0, H), range(0, W)):
            h_offset, w_offset = h*self.stride, w*self.stride
            rec_field = self.grad[:, :, h_offset:h_offset+KH, w_offset:w_offset+KW]
            self.grad[:, :, h_offset:h_offset+KH, w_offset:w_offset+KW] = (np.ones_like(rec_field) * dY[:, :, h, w].reshape(N, C, 1, 1))

        return (self.grad/(self.kernel_size[0]*self.kernel_size[0]))



class MaxPool2D(Function):
    def __init__(self, kernel_size=(2, 2), stride = 2):
        super().__init__()
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride

    def forward(self, X):
        # print("Max_pool forward started\n")
        N, C, H, W = X.shape
        KH, KW = self.kernel_size

        grad = np.zeros_like(X)
        out_shape = (N, C, 1 + int((H - KH)/self.stride), 1 + int((W - KW)/self.stride))
        Y = np.zeros(out_shape)
        # for n in range(N):
        for h, w in product(range(0, out_shape[2]), range(0, out_shape[3])):
            h_offset, w_offset = h*self.stride, w*self.stride
            rec_field = X[:, :, h_offset:h_offset+KH, w_offset:w_offset+KW]
            Y[:, :, h, w] = np.max(rec_field, axis=(2, 3))
            grad[:, :, h_offset:h_offset+KH, w_offset:w_offset+KW] = (X[:, :, h_offset:h_offset+KH, w_offset:w_offset+KW] >= np.max(rec_field, axis=(2, 3))[0][0])
            
            # for kh, kw in product(range(KH), range(KW)):
            #     #this contains the mask that will be multiplied to the max value error
            #     grad[:, :, h_offset+kh, w_offset+kw] = (X[:, :, h_offset+kh, w_offset+kw] >= Y[:, :, h, w])

        # storing the gradient
        self.local_grads['X'] = grad
        # print("Y", Y.shape)
        assert(not isnan(np.max(Y)))
        # print("Max_pool forward ended\n")
        return Y

    def backward(self, dY):
        # print("\nMax_pool backward started")
        KH, KW = self.kernel_size
        N, C, H, W = dY.shape
        for h, w in product(range(0, H), range(0, W)):
            h_offset, w_offset = h*self.stride, w*self.stride
            rec_field = self.local_grads['X'][:, :, h_offset:h_offset+KH, w_offset:w_offset+KW]
            self.local_grads['X'][:, :, h_offset:h_offset+KH, w_offset:w_offset+KW] = (rec_field * dY[:, :, h, w].reshape(N, C, 1, 1))
        
        assert(not isnan(np.max(self.local_grads['X'])))
        # print("Max_pool backward ended\n")
        return self.local_grads['X']

    def calculate_local_grads(self, X):
        
        return self.local_grads

class Flatten(Function):
    def forward(self, X):
        # print("\nFlatten forward started")
        self.cache['shape'] = X.shape
        n_batch = X.shape[0]
        X = X.reshape(-1, n_batch)
        # print("X", X.shape)
        assert(not isnan(np.max(X)))
        # print("Flatten forward ended\n")
        return X

    def backward(self, dY):
        # print("\nFlatten backward started")
        dY = dY.reshape(self.cache['shape'])
        assert(not isnan(np.max(dY)))
        return dY

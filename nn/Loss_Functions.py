import numpy as np
from .abstract_classes import Function

class Loss(Function):
    def forward(self, X, Y):
        '''
        Computes the loss of X with respect to  Y according to certain relation.
        Args:
            X: numpy.ndarray of shape (n_batch, n_dim).
            Y: numpy.ndarray of shape (n_batch, n_dim).
        Where:
            X : input numpy array , Y : output array
            n_batch : batch size.
            n_dim : signle input size
        '''
        pass

    def backward(self):
        '''
        Backward prob for the loss function which is considered the final layer of an architecture.
        no input is needed for the backward pass.

        Returns:
            gradX: numpy.ndarray of shape (n_batch, n_dim). Local gradient of the loss.
        '''
        return self.grad['X']

    def calculate_local_grads(self, X, Y):
        '''
        Local gradient with respect to X at (X, Y).
        Args:
            X: numpy.ndarray of shape (n_batch, n_dim).
            Y: numpy.ndarray of shape (n_batch, n_dim).
        Returns:
            gradX: numpy.ndarray of shape (n_batch, n_dim).
        '''
        pass

class CrossEntropyLoss(Loss):
    def forward(self, X, y):
        '''
        Computes the cross entropy loss of x with respect to y.
        Args:
            X: numpy.ndarray of shape (n_batch, n_dim).
            y: numpy.ndarray of shape (n_batch, 1). Should contain class labels
                for each data point in x.
        Returns:
            crossentropy_loss: numpy.float. Cross entropy loss of x with respect to y.
        '''
        # calculating crossentropy
        exp_x = np.exp(X)
        probs = exp_x/np.sum(exp_x, axis=1, keepdims=True)
        log_probs = -np.log([probs[i, y[i]] for i in range(len(probs))])
        crossentropy_loss = np.mean(log_probs)

        # caching for backprop
        self.cache['probs'] = probs
        self.cache['y'] = y

        return crossentropy_loss

    def calculate_local_grads(self, X, Y):
        probs = self.cache['probs']
        ones = np.zeros_like(probs)
        for row_idx, col_idx in enumerate(Y):
            ones[row_idx, col_idx] = 1.0

        grads = {'X': (probs - ones)/float(len(X))}
        return grads

class MeanSquareLoss(Loss):
    def forward(self, X, Y):

        '''        
        Computes the mean square error of X with respect to Y.
        Args:
            X: numpy.ndarray of shape (n_batch, n_dim).
            Y: numpy.ndarray of shape (n_batch, n_dim).
        Returns:
            mse_loss: numpy.float. Mean square error of x with respect to y.
        '''

        # calculating loss
        sum = np.sum((X - Y) ** 2, axis=1, keepdims=True)
        mse_loss = np.mean(sum)
        return mse_loss

    def calculate_local_grads(self, X, Y):
        '''
        Local gradient with respect to X at (X, Y).
        Args:
            X: numpy.ndarray of shape (n_batch, n_dim).
            Y: numpy.ndarray of shape (n_batch, n_dim).
        Returns:
            gradX: numpy.ndarray of shape (n_batch, n_dim). Gradient of MSE wrt X at X and Y.
        '''
        grads = {'X': 2 * (X - Y) / X.shape[0]}
        return grads



class L1_norm(Loss):
    def forward(self,X,y):
        sum = np.sum((X - Y), axis=1, keepdims=True)
        mae_loss = sum / X.shape[0]
        return mae_loss


    def calculate_local_grads(self,X,Y):
        grads = {'X': 1}
        return grads


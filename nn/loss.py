import numpy as np
from abstract_classes import Function

class Loss(Function):
    def forward(self, Y_hat, Y):
        pass
    def calculate_local_grads(self, Y_hat, Y):
        pass
    def backward(self):
        return self.local_grads["x"]
        
class MeanSquareLoss(Loss):
  def forward(self, Y_hat, Y):
    return ((Y_hat-Y)**2).mean(axis=1, keepdims=True)
  
  def calculate_local_grads(self, Y_hat, Y):
    # dL_dYhat = (2 * (Y_hat - Y))
    # m_examples = dL_dYhat.shape[0]
    # return {'x': dL_dYhat / m_examples }
    return {'x' : 2*(Y_hat - Y).mean(axis=1, keepdims=True) }
  

class CrossEntropyLoss(Loss):
    def forward(self, Y_hat, Y):
        """
            new

            yhat = (ndim, nbatch)
            y = (1, nbatch)
        """
        # calculating crossentropy
        probs = np.exp(Y_hat)/np.sum(Y_hat, axis=0, keepdims=True) # (ndim, nbatch)

        log_probs = -np.log( probs )

        #  ........... Problem ...............
        # Y is inf because y hat at the begin is very big (range 8k) so e^8k = inf 
        loss = Y * log_probs

        print("Dims", loss.shape)
        print(Y)
        print(probs)
        crossentropy_loss = np.mean(log_probs) # avrage on both axis 0 & axis 1 ()

        # caching for backprop
        self.cache['probs'] = probs
        self.cache['y'] = Y

        return crossentropy_loss

    def local_grad(self, X, Y):
        probs = self.cache['probs']
        ones = np.zeros_like(probs)
        for row_idx, col_idx in enumerate(Y):
            ones[row_idx, col_idx] = 1.0

        grads = {'X': (probs - ones)/float(len(X))}
        return grads


class NLLLoss(Loss):
    """
            yhat = (ndim, nbatch)
            y = (1, nbatch)
    """

    def forward(self, Y_hat, Y):
        
        logs = [-np.log( Y_hat[Y[i]] ) for i in range

        print("===============================")
        print(logs.shape)
            
            


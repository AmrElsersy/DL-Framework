import numpy as np
from abstract_classes import Function

class Loss(Function):
    def __init__(self,*args, **kwargs):
        super().__init__()
    
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
        exp_x = np.exp(Y_hat)
        probs = exp_x / np.sum(exp_x, axis=0, keepdims=True)

        log_probs = -np.log(probs)

        #  ........... Problem ...............
        # Y is inf because y hat at the begin is very big (range 8k) so e^8k = inf 

        crossentropy_loss = np.mean(log_probs,axis=0, keepdims=True) # avrage on both axis 0 & axis 1 ()
        #print("Dims", probs.shape)
        print('Label =',Y)
        print('Prediction = ',np.argmax(probs,axis=0))

        # caching for backprop
        self.cache['probs'] = probs
        self.cache['y'] = Y

        return crossentropy_loss

    def calculate_local_grads(self, X, Y):
        probs = self.cache['probs']
        b = np.zeros((probs.shape[1],probs.shape[0]))
        b[np.arange(Y.shape[1]),Y] = 1
        b = b.T
        probs = np.subtract(probs,b) / float(Y.shape[0])
        #probs =  probs.mean(axis=1,keepdims=True)
        return {'x':probs*X}

            
            


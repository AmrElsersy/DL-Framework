from abstract_classes import *


class Dense(Layer):

  def __init__(self,indim,outdim,*args, **kwargs):
    super().__init__()
    self.init_weights(indim,outdim)

  def init_weights(self,indim, outdim):
    """
      indim : x feature dimentions
      outdim : num of neurons in the layer
      w dims:
        (features x output_layar)
      b dims:
        (output layer x 1)
    """
   # xavier weight initialization
    self.weights['w'] = np.ones((indim,outdim)) * np.sqrt( 2/(indim+outdim) )
    self.weights['b'] = np.zeros((outdim, 1))

  def forward(self,X):

    # output dims = (output_layer x feeatures) . (features x batch_size) = (output_layer x batch_size)
    output = np.dot(self.weights['w'].T ,X) + self.weights['b']
    self.cache['x'] = X
    self.cache['output'] = output

    return output

  def backward(self,global_grad):
    dX = np.dot(self.local_grads['x'], global_grad )
    # dW dims = W dims .. because we have to calculate w = w - lr * dW
    # note that dW is global gradient .... but the local gradient (dY/dw) has a different dims as it is a function of the input
    dW = np.dot(np.array(self.local_grads['w']) , global_grad.T )
    # same as dW above
    db = np.sum(global_grad, axis = 0, keepdims = True)
    self.weights_global_grads = {'w': dW, 'b': db}
    return dX

  def calculate_local_grads(self, X):
    grads = {}
    grads['x'] = self.weights['w']
    grads['w'] = X.mean(axis = 1, keepdims = True) # mean for batch training
    grads['b'] = np.ones_like(self.weights['b'])
    return grads

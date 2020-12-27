from abstract_classes import *

class Dense(Layer):

  def __init__(self,indim,outdim,*args, **kwargs):
    super().__init__()
    self.init_weights(indim,outdim)

  def init_weights(self,indim, outdim):
   # xavier weight initialization
    self.weights['w'] = np.ones((indim,outdim)) * np.sqrt( 2/(indim+outdim) )
    self.weights['b'] = np.zeros((1,outdim))

  def forward(self,X):

    output = np.dot(X , self.weights['w']) + self.weights['b']
    self.cache['x'] = X
    self.cache['output'] = output

    return output

  def backward(self,dY):
    print(self.local_grads)
    dX = np.dot(dY,self.local_grads['x'].T)
    dW = np.dot(np.array(self.local_grads['w']).T,dY)
    db = np.sum(dY, axis = 0, keepdims = True)
    self.weights_global_grads = {'w': dW, 'b': db}
    print(self.weights_global_grads)
    return dX

  def calculate_local_grads(self, X):
    grads = {}
    grads['x'] = self.weights['w']
    grads['w'] = X
    grads['b'] = np.ones_like(self.weights['b'])
    return grads







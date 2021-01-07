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
    self.weights['w'] = np.random.randn(indim,outdim) * np.sqrt( 2/(indim+outdim) )
    self.weights['b'] = np.zeros((outdim, 1))

  def forward(self,X):
    output = np.dot(self.weights['w'].T ,X) + self.weights['b']
    self.cache['x'] = X
    self.cache['output'] = output

    return output

  def backward(self,global_grad):
    """
      compute backward probagation: multiply the global gradient with the local gradient with respect to each input (W,b,X)
      args:
        global_grad: global gradient of the next layer(represent dL/dX_of_next_layer) 
          dims: (output_nuorons_of_next_layer, batch_size)
      return:
        global gradient to backprobagate to the prev layer
          dims: (output_nuorons_of_current_layer, batch_size)
    """
    batch_size = global_grad.shape[1]


    dX = np.dot(self.local_grads['x'], global_grad )

    # ========= dW dim ==========
    # dW dims = W dims .. because we have to calculate w = w - lr * dW
    # note that dW is global gradient .... but the local gradient (dY/dw) has a different dims as it is a function of the input
    # dW(x_features, output) = dw_local(x_features, batch) * global.T(batch, output)

    # ========= / batch_size .. avarage over examples =========
    # devide by batch size because avarage is calculated due to matrix multiplication of the batch raw in dw_local & batch column in global_grad.T
    # so we need to devide because the matrix mul is a sum
    dW = np.dot(np.array(self.local_grads['w']) , global_grad.T ) / batch_size
    db = np.sum(global_grad, axis = 1, keepdims = True) / batch_size

    self.weights_global_grads = {'w': dW, 'b': db}

    # =============== PRINT ====================
    # print("global=",global_grad.shape, " ..dX=",dX.shape, " .. dW_glbal=",dW.shape," .. dW_local=",np.array(self.local_grads['w']).shape)

    # return the global gradient with respect to the input(the output of the prev layer)
    return dX

  def calculate_local_grads(self, X):
    grads = {}
    grads['x'] = self.weights['w']
    grads['w'] = X
    grads['b'] = np.ones_like(self.weights['b'])
    return grads

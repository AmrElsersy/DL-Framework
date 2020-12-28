from abstract_classes import *
from model import Model
from optim import *
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from matplotlib import style
from threading import Thread
import time
import sys

class Dense(Layer):

  def __init__(self,indim,outdim,*args, **kwargs):
    super().__init__()
    self.init_weights(indim,outdim)

  def init_weights(self,indim, outdim):
   # xavier weight initialization
    self.weights['w'] = np.ones((indim,outdim)) * np.sqrt( 2/(indim+outdim) )
    self.weights['b'] = np.zeros((1,outdim))

  def forward(self,X):
    # print(self.weights['b'].shape)
    output = np.dot(self.weights['w'].T ,X) + self.weights['b']
    self.cache['x'] = X
    self.cache['output'] = output

    return output

  def backward(self,dY):
    dX = np.dot(dY,self.local_grads['x'].T)
    dW = np.dot(np.array(self.local_grads['w']).T,dY)
    db = np.sum(dY, axis = 0, keepdims = True)
    self.weights_global_grads = {'w': dW, 'b': db}
    return dX

  def calculate_local_grads(self, X):
    grads = {}
    grads['x'] = self.weights['w']
    grads['w'] = X
    grads['b'] = np.ones_like(self.weights['b'])
    return grads

class MeanSquareLoss(Function):
  def forward(self, X, Y):
    return ((X-Y)**2).mean(axis=1, keepdims=True)
  def calculate_local_grads(self, Y_hat, Y):
    # dL_dYhat = (2 * (Y_hat - Y))
    # m_examples = dL_dYhat.shape[0]
    # return {'x': dL_dYhat / m_examples }
    return {'x' : 2*(Y_hat - Y).mean(axis=1, keepdims=True) }
  def backward(self):
    return self.local_grads["x"]

model = Model()
model.add(Dense(3,2))
# model.add(Dense(2,1))
model.set_loss(MeanSquareLoss())
optim = GradientDecent(model.parameters(),0.001)

# x = np.array([1,2,3], dtype=np.float).reshape(1,-1)
# y = np.array([2,4,6], dtype=np.float).reshape(1,-1)

x = np.array([ [1, 2, 3],
  [4,5,6],
  [7,8,9],
  [1,0,0]
],dtype=np.float32).T
y = np.array([6, 15, 24, 3], dtype=np.float32).reshape(1,-1)

# print(y.shape)
# print(y)

"""
[[1. 4. 7. 1.]
 [2. 5. 8. 0.]
 [3. 6. 9. 0.]]  .... shape (3, 4)

[[ 6. 15. 24., 3 ]] .... shape  (1, 4)
"""

# xx = x[:,1].reshape(-1,1)
# print(xx, xx.shape)

# print(x[:, 0].reshape(x.shape[0], 1).shape )
# print(y[:, 0].reshape(y.shape[0], 1).shape )
# exit()
epochs = 1000
for epoch in range(epochs):
    optim.zero_grad()
    y_hat = model.forward( x[:, 1].reshape(-1,1) )
    l = model.loss(y_hat, y[:, 1].reshape(-1,1) )
    model.backward()
    optim.step()
    print("y_hat= ", y_hat, " ... Loss = ", l)
    print("w=",model.layers[0].weights, " ... dw= ", model.layers[0].weights_global_grads["w"]," ... db= ", model.layers[0].weights_global_grads["b"])
    print("==========================================================================================================================================")

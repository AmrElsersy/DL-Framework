from abstract_classes import *
from model import Model
from optim import *

class Dense(Layer):

  def __init__(self,indim,outdim,*args, **kwargs):
    super().__init__()
    self.init_weights(indim,outdim)
    self.init_weights(indim, outdim)

  def init_weights(self,indim, outdim):
   # xavier weight initialization
    self.weights['w'] = np.ones((indim,outdim)) * np.sqrt( 2/(indim+outdim) )
    self.weights['b'] = np.zeros((1,outdim))

  def forward(self,X):
    output = X * self.weights['w'] + self.weights['b']

    self.cache['x'] = X
    self.cache['output'] = output
    return output

  def backward(self,dY):
    print(self.local_grads)
    dX = dY * self.local_grads['x'].T
    dW = np.array(self.local_grads['w']).T * dY
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

class MeanSquareLoss(Function):
  def forward(self, X, Y):
    return ((X-Y)**2).mean()
  def calculate_local_grads(self, X, Y):
    return {'x': 2 * (X - Y)}
  def backward(self):
    return self.local_grads["x"]
def relu(x):
    return x*(x > 0)
def relu_prime(x):
    return 1*(x > 0)
class ReLU(Function):
    def forward(self, X):
        return relu(X)
    def backward(self, dY):
        return dY * self.local_grads['X']
    def calculate_local_grads(self, X):
        grads = {'X': relu_prime(X)}
        return grads


x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
y = np.array([2, 4, 6, 8, 6], dtype=np.float32)

model = Model()
# y = w * x
model.add(Dense(1,1))
model.add(ReLU())
model.set_loss(MeanSquareLoss())
optim = GradientDecent(model.parameters(), 0.01)
epochs = 3


for epoch in range(epochs):
    y_hat = model.forward(x)
    l = model.loss(y_hat, y)
    model.backward()
    optim.step()
    print("y_hat= ", y_hat, " ... Loss = ", l)
    print("w=",model.layers[0].weights["w"], " ... dw= ", model.layers[0].weights_global_grads["w"])
    print("b=",model.layers[0].weights["b"], " ... db= ", model.layers[0].weights_global_grads["b"])
    print("==============================")
    # optim.zero_grad()


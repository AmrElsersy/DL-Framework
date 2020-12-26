from abstract_classes import *
from model import Model
from optim import *

class Dense(Layer):

  def __init__(self,indim,outdim):
    super().__init__()
    self.init_weights(indim,outdim)

  def init_weights(self,indim, outdim):
  # xavier weight initialization
    self.weights['w'] = np.random.randn(indim,outdim) * np.sqrt( 2/(indim+outdim) )
    self.weights['b'] = np.zeros(outdim)

  def forward(self,X):

    output = np.dot(X , self.weights['w']) + self.weights['b']
    self.cache['x'] = X
    self.cache['output'] = output

    return output

  def backward(self,dY):

    dX = np.dot(dY,self.local_grads['x'].T)
    X  = self.cache['x']
    dW = np.dot(self.local_grads['w'],dY)
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
    sum = np.sum((X - Y) ** 2, axis=1, keepdims=True)
    mse_loss = np.mean(sum)
    return mse_loss

  def calculate_local_grads(self, X, Y):
    grads = {'x': 2 * (X - Y) / X.shape[0]}
    return grads

  def backward(self):
    return self.local_grads["x"]

data = [
[2,14],
[3,16],
[55,120],
[100, 210],
[200, 410]
]



model = Model()
# y = w * x
model.add(Dense(1,1))
model.set_loss(MeanSquareLoss())

optim = GradientDecent(model.parameters(), 0.001)

epochs = 1


for epoch in range(epochs):
  for d in data:
    x = d[0]
    y = d[1]

    print(x,y)
    y_hat = model.forward(x)
    l = model.loss(y_hat, y)
    model.backward()

    optim.step()

    print("y_hat= ", y_hat, " ... Loss = ", l)
    print("w=",model.layers[0].weights["w"], " ... dw= ", model.layers[0].weights_global_grads["w"])
    print("==============================")

    optim.zero_grad()


from abstract_classes import *
from model import Model
from optim import *
from Linear import *
from loss import *

model = Model()
model.add(Dense(1,1))
model.set_loss(MeanSquareLoss())
optim = GradientDecent(model.parameters(),0.01)

x = np.array([1,2,3,4,5], dtype=np.float).reshape(1,-1)
y = np.array([2,4,6,8,10], dtype=np.float).reshape(1,-1)
print(x.shape, y.shape)

epochs = 400
for epoch in range(epochs):
    y_hat = model.forward(x)
    l = model.loss(y_hat, y)
    model.backward()
    optim.step()
    print("y_hat= ", y_hat, " ... Loss = ", l)
    print("================================ ")

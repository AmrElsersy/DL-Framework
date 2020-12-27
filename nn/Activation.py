from .Activation_Functions import *
from .abstract_classes import Function

class ActivationFn(Function):
    def backward(self, dy):
        return dy * self.local_grads['x']


class Sigmoid(ActivationFn):

    def forward(self, x):
        return sigmoid(x)

    def calculate_local_grads(self, x):
        return {'x': sigmoid_derivative(x)}


class ReLU (ActivationFn):
    def forward(self, x):
        return relu(x)

    def calculate_local_grads(self, x):
        return {'x': relu_derivative(x)}


class Leaky_ReLU (ActivationFn):
    def forward(self, x):
        return leaky_relu(x)
    def calculate_local_grads(self, x):
        return {'x': leaky_relu_derivative(x)}


class Tanh (ActivationFn):
    def forward(self, x):
        return tanh(x)
    def calculate_local_grads(self, x):
        return {'x': tanh_derivative(x)}

class SoftMax (ActivationFn):
    def forward(self, x):
        return softMax(x)
    def backward(self, dy):
        pass
    def calculate_local_grads(self, x):
        pass

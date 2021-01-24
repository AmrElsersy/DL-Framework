from abstract_classes import Function

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.abs(x)))

def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))

def relu (x) :
    return x*(x > 0)

def relu_derivative(x):
    return 1*(x > 0)

def leaky_relu (x,alpha):
    return x*(x > 0) + alpha*x*(x <= 0)

def leaky_relu_derivative(x,alpha):
    return 1*(x > 0) + alpha*(x <= 0)

def softMax(x):
    max_1 = np.max(x, axis=0, keepdims=True)
    max_1 = np.subtract(x, max_1)
    exp_x = np.exp(max_1)
    value = np.divide(exp_x, np.sum(exp_x, axis=0, keepdims=True))
    return value

def tanh (x):
    return np.tanh(x)

def tanh_derivative (x):
    return 1-(np.tanh(x) ** 2)

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
    def forward(self, x, alpha):
        self.alpha = alpha
        return leaky_relu(x, self.alpha)
    def calculate_local_grads(self, x):
        return {'x': leaky_relu_derivative(x, self.alpha)}


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

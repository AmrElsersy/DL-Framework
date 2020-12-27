import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))

def relu (x) :
    if x > 0 :
        return x
    return 0

def relu_derivative(x):
    if x > 0:
        return 1
    return 0

def leaky_relu (x,alpha):
    if x > 0:
        return x
    return alpha * x

def leaky_relu_derivative(x,alpha):
    if x > 0:
        return 1
    return alpha

def softMax (x):
    exp_x = np.exp(x)
    value = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return value

def tanh (x):
    return np.tanh(x)

def tanh_derivative (x):
    return 1-(np.tanh(x) ** 2)


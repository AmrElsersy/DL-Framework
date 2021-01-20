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
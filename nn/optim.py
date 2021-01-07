from abstract_classes import Function, Layer
import numpy as np


class Optimizer:
    """
        Optimizer for weight update process with different Gradient Decent algorithms
    """
    def __init__(self, parameters, learning_rate):
        # model layers
        self.parameters = parameters
        self.lr = learning_rate

    def step(self):
        """
            1 Step of weights update
        """

        for layer in self.parameters:
            # get layer's weights & d_weights  dictionaries and optimize them by various types of optimization algorithms
            for key, value in layer.weights.items():
                layer.weights[key] = self.optimize(layer.weights[key], layer.weights_global_grads[key])

    def zero_grad(self):
        for layer in self.parameters:
            for key, value in layer.weights.items():
                layer.weights_global_grads[key] = 0
        

    def optimize(self, w, dw):
        """
            Optimization Equation for different types of gradient decent 
        """
        return w

class GradientDecent(Optimizer):
    def optimize(self, w, dw):
        # dw = np.mean(dw, axis=1, keepdims=True)
        dw = dw / np.max(dw)
        w = w - self.lr * dw
        return w

class SGD(Optimizer):
    """
        Stochastic Gradient Decent 
    """
    def optimize(self, w, dw):
        # get best learning rate
        self.lr = np.argmin(w - self.lr * dw)
        w = w - self.lr * dw
        return w

class MomentumGD(GradientDecent):
    """
        Gradient Decent algorithm based on Avarage of weights grads for better estimation of grads
    """
    def __init__(self, parameters, learning_rate, beta):
        super().__init__(parameters, learning_rate)

        # avarage weight parameter
        self.beta = beta

        # make V_dw as a list of dicts with list_size = layes num
        self.V_dW = [ {} for i in self.parameters]

        # zero initialization
        for i, layer in enumerate(self.parameters):
            for key, _ in layer.weights.items():
                self.V_dW[i][key] = 0
            
    def step(self):
        for i, layer in enumerate(self.parameters):
            for key, _ in layer.weights.items():
                # V_dW = B * V_dW_prev + (1-B) * dW
                self.V_dW[i][key] = self.beta * self.V_dW[i][key] + (1-self.beta) * layer.weights_global_grads[key]
                layer.weights[key] = self.optimize(layer.weights[key], self.V_dW[i][key])
                

class Adam(GradientDecent):
    """
        Gradient Decent algorithm based on Avarage of weights gradients + RMS of gradients
    """
    def __init__(self, parameters, learning_rate, beta_Vdw, beta_Sdw):
        super().__init__(parameters, learning_rate)

        # avarage weight parameter
        self.beta_Vdw = beta_Vdw
        self.beta_Sdw = beta_Sdw
        self.epsilon = 1e-5

        # make V_dw as a list of dicts with list_size = layes num
        self.V_dW = [ {} for i in self.parameters]

        # zero initialization
        for layer in self.parameters:
            for key, _ in layer.weights.items():
                self.V_dW[i][key] = 0

        # as follow .. zero initialization
        self.S_dw = self.V_dW

    def step(self):
        for i, layer in enumerate(self.parameters):
            for key, _ in weights.items():
                # V_dW = B1 * V_dW_prev + (1-B) * dW
                self.V_dW[i][key] = self.beta_Vdw * self.V_dW[i][key] + (1-self.beta_Vdw) * layer.weights_global_grads[key]

                # S_dw = B2 * S_dw_prev + (1-B2) * dW^2
                self.S_dw[i][key] = self.beta_Sdw * self.S_dw[i][key] + (1-self.beta_Sdw) * (layer.weights_global_grads[key] **2 ) 

                # w = w - lr * V_dw / (sqrt(S_dw) + E)
                layer.weights[key] = self.optimize(layer.weights[key], self.V_dW[i][key] / (np.sqrt(self.S_dw[i][key]) + self.epsilon) )        


class StepLR_Schedular:
    def __init__(self, optimizer, step_size, gamma):
        self.optimizer = optimizer 
        self.step_count = 0
        self.step_size = step_size
        self.gamma = gamma
    
    def step(self):

        # increment steps num to keep track of steps for LR_Schedular
        self.step_count = self.step_count + 1

        if self.step_count % self.step_size == 0:
            self.optimizer.lr = self.optimizer.lr * self.gamma

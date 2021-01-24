from abstract_classes import Function, Layer
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib import style
import sys
import time
from threading import Thread
from visualize import graph

class Model():
    def __init__(self):
        # list of layers & functions
        self.layers = []
        # loss function
        self.loss_function = None
        # train / eval mode 
        self.is_train_mode = False
        self.trainingLoss = []
        self.validationLoss = None
        self.static_graph = False
        self.th = Thread(target=self.live_graph, daemon=True)

    def live_graph(self):
        """
        Draw a graph for training and validation loss\n
        [Required] trainingLoss: Training loss array,\t
        [optional] validationLoss: Validation loss array,
        """
        # style.use('fivethirtyeight')
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        def animate(i):
            ax1.cla()
            if self.validationLoss is not None:
                ax1.plot(np.array(range(len(self.trainingLoss))) + 1, self.validationLoss, label="Validation loss")
                # print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(len(self.trainingLoss), self.trainingLoss[-1], self.validationLoss[-1]))
            else:
                # print('Epoch: {} \tTraining Loss: {:.6f}'.format(len(self.trainingLoss), self.trainingLoss[-1]))
                ax1.plot(np.array(range(len(self.trainingLoss))) + 1, self.trainingLoss, label="Training loss")
            plt.legend(loc='best')
            plt.xlabel('Iteration number')
            plt.ylabel('Loss')
        anim = FuncAnimation(fig, animate, interval=1000)
        plt.tight_layout()
        plt.show()
    
    def graph(self):
        self.static_graph = (not self.static_graph)
    
    def add(self, layer):
        # all layers should be subclasss of Function class
        if not isinstance(layer, Function):
            raise Exception("The layer/function"+ str(layer.__doc__()) + " is not supported")

        self.layers.append(layer)

    def set_loss(self, loss_function):
        self.loss_function = loss_function


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)         
        return x

    def loss(self, predictions, labels):
        if not self.loss:
            raise Exception("Model has not a Loss Fn")
        
        loss = self.loss_function(predictions, labels)
        self.trainingLoss.append(round(loss, 5))
        if self.static_graph:
            graph(self.trainingLoss)
        return loss
        

    def backward(self):
        global_grad = self.loss_function.backward()
        for layer in reversed(self.layers):
            global_grad = layer.backward(global_grad)

    def __call__(self, x):
        return self.forward(x)

    def train_mode(self):
        self.is_train_mode = True

    def eval_mode(self):
        self.is_train_mode = False

    def parameters(self):
        return [layer for layer in self.layers if isinstance(layer, Layer)]

    def startGraph(self):
        self.trainingLoss = []
        self.th.start()

    def stopGraph(self):
        self.th.join()

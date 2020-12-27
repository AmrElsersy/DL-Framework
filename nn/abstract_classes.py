import numpy as np

class Function:
    def __init__(self, *args, **kwargs):
        # caching inputs
        self.cache = {}
        # caching gradients
        self.local_grads = {}

    # *args, **kwargs .. can pass any num/type of args to the function
    def forward(self, *args, **kwargs):
        """
            Forward Probagation - Compute function output
            return:
                Function output
        """
        pass

    def backward(self, global_grad):
        """
            Backprobagation - Compute global gradient
            args:
                global_grad: previous global gradient to calculate the new global gradient
            return:
                New global gradient to be backprobagated to the previous function/layer
        """
        pass

    def calculate_local_grads(self, *args, **kwargs):
        """
            Local gradients - d_output / d_inputs
            if the function is function of many inputs, calculate the jecopian vector (gradient of each one of the inputs)
            and store it in the grads
            return:
                local gradients of the function/layer
        """
        pass

    # Operator () overloading .. get called when calling "obj()"
    def __call__(self, *args, **kwargs):
        output = self.forward(*args, **kwargs)

        # gradient history for backprobagation
        self.local_grads = self.calculate_local_grads(*args, **kwargs)

        return output
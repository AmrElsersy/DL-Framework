from abstract_classes import Function, Layer


class Model():
    def __init__(self):
        # list of layers & functions
        self.layers = []
        # loss function
        self.loss_function = None
        # train / eval mode 
        self.is_train_mode = False

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
        return self.loss_function(predictions, labels)
        

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
        return self.layers
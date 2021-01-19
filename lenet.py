from dataset import MNIST_dataset, Dataset, Data_Loader
from model import Model
from Linear import Dense
from optim import GradientDecent, MomentumGD,  Adam, StepLR
from activations import ReLU, Sigmoid
from loss import CrossEntropyLoss
from utils import save_weights, load_weights
from cnn import *
import time

# MNIST Dataset
batch_size = 1
dataset = MNIST_dataset("train.csv")
dataloader = Data_Loader(dataset, batch_size)

# LeNet
lenet = Model()

lenet.add(conv(1, 6, 5, padding=2))
lenet.add(Sigmoid())
lenet.add(MaxPool2D(kernel_size=2))
lenet.add(conv(6, 16, 5))
lenet.add(Sigmoid())
lenet.add(MaxPool2D(kernel_size=2))
lenet.add(Flatten())

lenet.add(Dense(16*5*5, 120))
lenet.add(Sigmoid())
lenet.add(Dense(120, 84))
lenet.add(Sigmoid())
lenet.add(Dense(84, 10))

lenet.set_loss(CrossEntropyLoss())

optimizer = Adam(lenet.parameters(), learning_rate=0.01)
lr_schedular = StepLR(optimizer, step_size=1, gamma=0.1)

epochs = 10
for epoch in range(epochs):
    i = 0
    for image, label in dataloader:
        # if i == 1700:
        #     break
        image = image/255
        image = image.reshape(batch_size, 1, 28, 28)
        i = i + 1
        print("Iteration no.", i)
        predicted = lenet(image)
        loss = lenet.loss(predicted, label)
        lenet.backward()
        optimizer.step()
        print("loss= ", loss)
        #time.sleep(0.1)
        print("===========")
    lr_schedular.step()
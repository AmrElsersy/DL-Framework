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
batch_size = 2
dataset = MNIST_dataset("train.csv")
dataloader = Data_Loader(dataset, batch_size)

# LeNet
lenet = Model()

lenet.add(conv(1, 6, 5, padding=2))
lenet.add(ReLU())
lenet.add(AvgPool2D(kernel_size=2))
lenet.add(conv(6, 16, 5))
lenet.add(ReLU())
lenet.add(AvgPool2D(kernel_size=2))
lenet.add(Flatten())

lenet.add(Dense(16*5*5, 120))
lenet.add(ReLU())
lenet.add(Dense(120, 84))
lenet.add(ReLU())
lenet.add(Dense(84, 10))

lenet.set_loss(CrossEntropyLoss())

optimizer = GradientDecent(lenet.parameters(), learning_rate=0.1)

epochs = 10
for epoch in range(epochs):
    i = 0
    for image, label in dataloader:
        # if i == 1:
        #     break

        for batch_idx in range(image.shape[1]):
            img = image[:, batch_idx].reshape(1, 28, 28)
            img = img/255
            images.append(img)
        images = np.asarray(images)

        i = i + 1
        print("Iteration no.", i)
        predicted = lenet(images)
        loss = lenet.loss(predicted, label)
        lenet.backward()
        optimizer.step()
        print("loss= ", loss)
        #time.sleep(0.1)
        print("===========")

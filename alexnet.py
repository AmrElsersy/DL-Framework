from dataset import MNIST_dataset, Dataset, Data_Loader
from model import Model
from Linear import Dense
from optim import GradientDecent, MomentumGD,  Adam, StepLR
from activations import ReLU, Sigmoid
from loss import CrossEntropyLoss
from utils import save_weights, load_weights
from cnn import *
from PIL import Image
import cv2
import time
import numpy as np

# MNIST Dataset
batch_size = 1
dataset = MNIST_dataset("train.csv")
dataloader = Data_Loader(dataset, batch_size)



alexnet = Model()

alexnet.add(conv(1, 96, 11, padding=1, stride=4))
alexnet.add(ReLU())
alexnet.add(MaxPool2D(kernel_size=3))
alexnet.add(conv(96, 256, 5, padding=2))
alexnet.add(Sigmoid())
alexnet.add(MaxPool2D(kernel_size=3))
alexnet.add(conv(256, 384, 3, padding=1))
alexnet.add(Sigmoid())
alexnet.add(conv(384, 384, 3, padding=1))
alexnet.add(ReLU())
alexnet.add(conv(384, 256, 3, padding=1))
alexnet.add(ReLU())
alexnet.add(MaxPool2D(kernel_size=3))
alexnet.add(Flatten())
alexnet.add(Dense(6400, 4096))
alexnet.add(ReLU())
alexnet.add(Dense(4096, 4096))
alexnet.add(ReLU())
alexnet.add(Dense(4096, 10))

alexnet.set_loss(CrossEntropyLoss())

optimizer = GradientDecent(alexnet.parameters(), learning_rate=0.01)

epochs = 10

for epoch in range(epochs):
    i = 0
    for img, label in dataloader:
        # if i == 1700:
        #     break
        img = img.reshape(28, 28)
        img = np.asarray(img, dtype='int8')
        
        # cv_gray = cv2.CreateMat(28, 28, cv2.CV_32FC3)
        image = Image.fromarray(img)
        image = image.resize(size=(224, 224))
        image = np.asarray(image)

        print(image.shape)
        image = image/255
        image = image.reshape(batch_size, 1, 224, 224)
        i = i + 1
        print("Iteration no.", i)
        predicted = alexnet(image)
        loss = alexnet.loss(predicted, label)
        alexnet.backward()
        optimizer.step()
        print("loss= ", loss)
        #time.sleep(0.1)
        print("===========")

from dataset import MNIST_dataset, Dataset, Data_Loader
from model import Model
from Linear import Dense
from optim import GradientDecent, MomentumGD,  Adam, StepLR
from activations import ReLU, Sigmoid, softMax
from loss import CrossEntropyLoss
from utils import save_weights, load_weights
from evaluation import Evaluation
from visualize import img_viewer_examples
from cnn import *
import time
import numpy as np


batch_size = 1
dataset = MNIST_dataset("./datasets/train.csv")
dataloader = Data_Loader(dataset, batch_size)


model = Model()
model.add(conv(1, 4, 3, padding=1))
model.add(MaxPool2D(kernel_size=(2,2)))
model.add(ReLU())
model.add(conv(4, 8, 3, padding=1))
model.add(MaxPool2D(kernel_size=2))
model.add(ReLU())
model.add(Flatten())
model.add(Dense(8*7*7, 10))


model.set_loss(CrossEntropyLoss())
optimizer = Adam(model.parameters(), learning_rate = 0.01)
lr_schedular = StepLR(optimizer, step_size = 1, gamma=0.1)


epochs = 1
for epoch in range(epochs):
    i = 0
    for image, label in dataloader:
        image = image/255
        images = image.T.reshape(batch_size, 1, 28, 28)
        images = np.asarray(images)
        i = i + 1
        print("Iteration no.", i)
        predicted = model(images)
        loss = model.loss(predicted, label)
        model.backward()
        optimizer.step()
        print("loss= ", loss)
        # time.sleep(0.1)
        print("===========")
    lr_schedular.step()


test_dataset = MNIST_dataset("./datasets/train.csv")
dataloader_test = Data_Loader(test_dataset, batch_size)

e = Evaluation(10)

for image, label in dataloader_test:
    image = image/255
    images = image.T.reshape(batch_size, 1, 28, 28)
    images = np.asarray(images)
    predicted = model(images)
    probs = softMax(predicted)
    pred = np.argmax(probs,axis=0)
    e.add_prediction(pred[np.newaxis],label)
print("the confusion Matrix:\n",e.get_confusion_Matrix())
print("the Mean F1 Score:\n",e.evaluate())
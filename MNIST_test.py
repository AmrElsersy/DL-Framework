from dataset import MNIST_dataset, Dataset, Data_Loader
from model import Model
from Linear import Dense
from optim import GradientDecent, MomentumGD,  Adam, StepLR
from activations import ReLU,Sigmoid
from loss import CrossEntropyLoss
from utils import save_weights, load_weights
import numpy as np
import sys
import time
from evaluation import Evaluation
from activation_functions import *
# weights path

path = "ray2_weights.sav"

# MNIST Dataset
batch_size = 32
train_dataset = MNIST_dataset("train.csv")
test_dataset = MNIST_dataset("train.csv")
dataloader_train = Data_Loader(train_dataset, batch_size)
dataloader_test = Data_Loader(test_dataset, batch_size)

model = Model()
model.add(Dense(784, 90))
model.add(ReLU())
model.add(Dense(90, 45))
model.add(ReLU())
model.add(Dense(45, 10))

model.set_loss(CrossEntropyLoss())

optimizer = GradientDecent(model.parameters(), learning_rate = 0.01)
# optimizer = MomentumGD(model.parameters(), learning_rate = 0.01)
# optimizer = Adam(model.parameters(), learning_rate = 0.01)
# lr_schedular = StepLR(optimizer, step_size = 1, gamma=0.1)

# model = load_weights(path)

epochs = 1
# model.startGraph()
for epoch in range(epochs):
    i = 0
    for image, label in dataloader_train:
        # if i == 1700:
        #     break
        image = image/255
        i = i + 1
        print("Iteration no.", i)
        predicted = model(image)
        loss = model.loss(predicted, label)
        model.backward()
        optimizer.step()
        # print("loss= ", loss)
        print("===========")

e = Evaluation(10)

for image, label in dataloader_test:
    image = image/255
    predicted = model(image)
    probs = softMax(predicted)
    pred = np.argmax(probs,axis=0)
    e.add_prediction(pred[np.newaxis],label)
print("the confusion Matrix :",e.get_confusion_Matrix())
print("the Mean F1 Score =",e.evaluate())

# save_weights(model, path)

print("Enter any key to exit")
x = input()
# model.stopGraph()
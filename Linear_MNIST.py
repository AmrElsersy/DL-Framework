from dataset import MNIST_dataset, Dataset, Data_Loader
from model import Model
from Linear import Dense
from optim import GradientDecent, MomentumGD,  Adam, StepLR
from activations import ReLU, Sigmoid, softMax
from loss import CrossEntropyLoss
from utils import save_weights, load_weights
from cnn import *
from visualize import img_viewer_examples
from evaluation import Evaluation
import time
import numpy as np


# MNIST Dataset
batch_size = 32
dataset = MNIST_dataset("./datasets/train.csv")
dataloader = Data_Loader(dataset, batch_size)
test_dataset = MNIST_dataset("./datasets/train.csv")
dataloader_test = Data_Loader(test_dataset, batch_size)


image, labels = dataloader.__getitem__(1)
images = image.T.reshape(batch_size, 1, 28, 28)
images = np.asarray(images)


img_viewer_examples(images, labels.tolist()[0], greyscale= True)


model = Model()
model.add(Dense(784, 90))
model.add(ReLU())
model.add(Dense(90, 45))
model.add(ReLU())
model.add(Dense(45, 10))


model.set_loss(CrossEntropyLoss())
optimizer = Adam(model.parameters(), learning_rate = 0.01)
lr_schedular = StepLR(optimizer, step_size = 1, gamma=0.1)


# weights path
path = "./checkpoints/Linear_MINST_weights.sav"
# model = load_weights(path)

epochs = 6
for epoch in range(epochs):
    i = 0
    for image, label in dataloader:
        if epoch == 5:
            model.graph()
        image = image/255
        i = i + 1
        print("Iteration no.", i)
        predicted = model(image)
        loss = model.loss(predicted, label)
        model.backward()
        optimizer.step()
        # print("loss= ", loss)
        print("===========")
    lr_schedular.step()


# save_weights(model, path)

e = Evaluation(10)


for image, label in dataloader_test:
    image = image/255
    predicted = model(image)
    probs = softMax(predicted)
    pred = np.argmax(probs,axis=0)
    e.add_prediction(pred[np.newaxis],label)
print("the confusion Matrix:\n",e.get_confusion_Matrix())
print("the Mean F1 Score:\n",e.evaluate())

model1 = Model()
model1.add(Dense(784, 90))
model1.add(ReLU())
model1.add(Dense(90, 45))
model1.add(ReLU())
model1.add(Dense(45, 10))

model1.set_loss(CrossEntropyLoss())
optimizer1 = GradientDecent(model1.parameters(), learning_rate = 0.01)

epochs = 6
for epoch in range(epochs):
    i = 0
    for image, label in dataloader:
        if epoch ==5 and i == 624:
            model1.graph()
        image = image/255
        i = i + 1
        print("Iteration no.", i)
        predicted = model1(image)
        loss = model1.loss(predicted, label)
        model1.backward()
        optimizer1.step()
        # print("loss= ", loss)
        print("===========")

e1 = Evaluation(10)

for image, label in dataloader_test:
    image = image/255
    predicted = model1(image)
    probs = softMax(predicted)
    pred = np.argmax(probs,axis=0)
    e1.add_prediction(pred[np.newaxis],label)
print("the confusion Matrix:\n",e1.get_confusion_Matrix())
print("the Mean F1 Score:\n",e1.evaluate())

model2 = Model()
model2.add(Dense(784, 90))
model2.add(ReLU())
model2.add(Dense(90, 45))
model2.add(ReLU())
model2.add(Dense(45, 10))

model2.set_loss(CrossEntropyLoss())
optimizer2 = MomentumGD(model2.parameters(), learning_rate = 0.01)

epochs = 6
for epoch in range(epochs):
    i = 0
    for image, label in dataloader:
        if epoch == 5 and i == 624:
            model2.graph()
        image = image/255
        i = i + 1
        print("Iteration no.", i)
        predicted = model2(image)
        loss = model2.loss(predicted, label)
        model2.backward()
        optimizer2.step()
        # print("loss= ", loss)
        print("===========")

e2 = Evaluation(10)

for image, label in dataloader_test:
    image = image/255
    predicted = model2(image)
    probs = softMax(predicted)
    pred = np.argmax(probs,axis=0)
    e2.add_prediction(pred[np.newaxis],label)
print("the confusion Matrix:\n",e2.get_confusion_Matrix())
print("the Mean F1 Score:\n",e2.evaluate())





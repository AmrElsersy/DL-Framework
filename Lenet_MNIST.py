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
optimizer = GradientDecent(lenet.parameters(), learning_rate=0.01)

epochs = 1
for epoch in range(epochs):
    i = 0
    for image, label in dataloader:
        if i == 19999:
          model.graph()
        images = image.T.reshape(batch_size, 1, 28, 28)
        images = images/255
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

batch_size = 1
test_dataset = MNIST_dataset("./datasets/train.csv")
dataloader_test = Data_Loader(test_dataset, batch_size)

e = Evaluation(10)

for image, label in dataloader_test:
    image = image/255
    images = image.T.reshape(batch_size, 1, 28, 28)
    images = np.asarray(images)
    predicted = lenet(images)
    probs = softMax(predicted)
    pred = np.argmax(probs,axis=0)
    e.add_prediction(pred[np.newaxis],label)
print("the confusion Matrix:\n",e.get_confusion_Matrix())
print("the Mean F1 Score:\n",e.evaluate())
from dataset import Dataset, Data_Loader, MNIST_dataset
from model import Model
from Linear import Dense
from optim import GradientDecent, MomentumGD,  Adam, StepLR
from activations import ReLU,Sigmoid
from loss import CrossEntropyLoss
from utils import save_weights, load_weights
from cnn import *
import time

# wieghts path
path = "ray2_weights.sav"


# MNIST Dataset
batch_size = 1
dataset = MNIST_dataset("train.csv")
dataloader = Data_Loader(dataset, batch_size)

#exit()
#ob1.backward()




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


# optimizer = GradientDecent(model.parameters(), learning_rate = 0.01)
# optimizer = MomentumGD(model.parameters(), learning_rate = 0.01)
optimizer = Adam(model.parameters(), learning_rate = 0.01)
lr_schedular = StepLR(optimizer, step_size = 1, gamma=0.1)

#model = load_weights(path)

epochs = 1
for epoch in range(epochs):
    i = 0
    for image, label in dataloader:
        if i == 1000:
            break
        image = image/255
        image = image.T.reshape(batch_size,1,28,28)
        i = i + 1
        print("Iteration no.", i)
        predicted = model(image)
        loss = model.loss(predicted, label)
        model.backward()
        optimizer.step()
        print("loss= ", loss)
        # time.sleep(0.1)
        print("===========")
print(np.asarray(model.trainingLoss).reshape(-1, 100))
# save_weights(model, path)

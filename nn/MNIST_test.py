from dataset import Dataset, Data_Loader
from model import Model
from Linear import Dense
from optim import GradientDecent, SGD, Adam
from activations import ReLU
from loss import CrossEntropyLoss, NLLLoss

# MNIST Dataset
dataset = Dataset("train.csv")
dataloader = Data_Loader(dataset, 4)


model = Model()
model.add(Dense(784, 90))
model.add(ReLU())
model.add(Dense(90, 45))
model.add(ReLU())
model.add(Dense(45, 10))

model.set_loss(NLLLoss())

optimizer = GradientDecent(model.parameters(), learning_rate=0.01)


for image, label in dataloader:
    print(image.shape, label.shape)
    predicted = model(image)
    loss = model.loss(predicted, label)
    model.backward()
    optimizer.step()
    print("loss= ", loss)
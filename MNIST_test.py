from nn.dataset import Dataset, Data_Loader
from nn.model import Model
from nn.Linear import Dense
from nn.optim import *


# MNIST Dataset
dataset = Dataset("nn/train.csv")
dataloader = Data_Loader(dataset, 4)


model = Model()
model.add()



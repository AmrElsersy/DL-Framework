# Pyray2 Deep Learning Framework 

Pyray2 is a deep learning framework implemnted from scratch using python and libraries such as numpy, matplotlib, pandas.


### How to Install
```
 $ pip install pyray2
```

### How to use ?
**Check Colab:**
- [Train on MNIST with Linear Layers](https://colab.research.google.com/drive/1cDJJm6eL--yMaleA0yBactI1qPdslDun)
- [Train on MNIST with CNN](https://colab.research.google.com/drive/17Fqk_qaEaXLGSRWzSfkjs9GY51D3nGKZ?usp=sharing)
- [Lenet archeticture](https://colab.research.google.com/drive/1JIeUIpiJrc1Mh2Rc8mIsij-uqRTme04v?usp=sharing)
### Design
![DL_Framework_UML](https://user-images.githubusercontent.com/35613645/105637853-4ca68500-5e78-11eb-9dc3-33f68bfd1cdf.png)


### Supported Features
#### Layers
- Convolution 2D
- Dense 
- Max Pooling
- Avg Pooling
#### Activations
- Sigmoid
- RelU
- Softmax
- Tanh
#### Losses
- Mean Square Error
- Cross Entropy 

**How to define a Model:**
```python
# Example
model = Model()
model.add(conv(1, 4, 3, padding=1))
model.add(MaxPool2D(kernel_size=(2,2)))
model.add(ReLU())
model.add(Flatten())
model.add(Dense(8*7*7, 10))
model.set_loss(CrossEntropyLoss())
```
#### Optimizers
- Gradiant Decent
- Momentum 
- Adam
```python
optim = GradientDecent(model.parameters(), learning_rate = 0.01)
optim = MomentumGD(model.parameters(), learning_rate = 0.01)
optim = Adam(model.parameters(), learning_rate = 0.01)
lr_schedular = StepLR(optimizer, step_size = 1, gamma=0.1)
```
#### Visualization 
- View dataset's examples
- Plotting for loss
```python
# Show Dataset Examples
image, labels = dataloader.__getitem__(1)
images = image.T.reshape(batch_size, 1, width, height)
images = np.asarray(images)
img_viewer_examples(images, labels.tolist()[0], greyscale= True)

# Plotting Loss
model.graph()
# Training Loop ...
```
#### Evaluation
- Precision
- Recall
- F1 Score

```python
e = Evaluation(10)
for image, label in dataloader_test:
    predicted = .....
    pred = .....
    e.add_prediction(pred[np.newaxis],label)
print("the confusion Matrix :",e.get_confusion_Matrix())
print("the Mean F1 Score =",e.evaluate())
```
#### Dataset & Data Loader
- CSV dataset loading
- Batch loading by Data Loader
- Interface for MNIST & CIFAR-10 datasets
- Simillar to Pytorch Interface.

```python
batch_size = 4
dataset = MNIST_dataset("train.csv")
dataloader = Data_Loader(dataset, batch_size)
for image, label in dataloader:
	...
	...
```
#### Save / Load utils
- Saving / Loading model weights through .pth files.

```python
model = load_weights(path)
save_weights(model, path)
```
#### Architectures
- LeNet
- AlexNet

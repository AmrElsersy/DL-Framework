import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import torch
from torchvision import datasets
import torchvision.transforms as transforms

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# choose the training and test datasets
train_data = datasets.MNIST(root='data', train=True,
                            download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
                           download=True, transform=transform)

num_workers = 0
# how many samples per batch to load
batch_size = 32

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                          num_workers=num_workers)

dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()


def img_viewer_examples(images, labels, prediction=None, size=0, greyscale=False):
    batchSize = min(size, images.shape[0])

    if size == 0:
      batchSize = images.shape[0]

    # I CAN TAKE THE BATCH_SIZE from the images size/shape according the sent data type
    no_of_columns = round(math.sqrt(batchSize))
    no_of_rows = math.ceil(batchSize / no_of_columns)
    print("batch size {}, no_of_rows {}, no_of_columns {}".format(
        batchSize, no_of_rows, no_of_columns))
    fig = plt.figure(figsize=(no_of_columns*1.25, no_of_rows*1.5))
    # (width, height)
    for idx in np.arange(batchSize):
        ax = fig.add_subplot(no_of_rows, no_of_columns,
                             idx+1, xticks=[], yticks=[])
        if greyscale:
          ax.imshow(np.squeeze(images[idx]), cmap='gray')
        else:
          ax.imshow(np.squeeze(images[idx]))
        # print out the correct label for each image
        # .item() gets the value contained in a Tensor
        # WAIT FOR TASNEEM TO SEE THE RETURNED DATA TYPE
        if prediction is not None:
            ax.set_title("{} ({})".format(str(prediction[idx].item()), str(labels[idx].item())),
                         color=("green" if prediction[idx] == labels[idx] else "red"))
        else:
            ax.set_title(str(labels[idx].item()))

img_viewer_examples(images, labels, greyscale=True)

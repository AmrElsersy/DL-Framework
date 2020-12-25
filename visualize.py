import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt


# ******************************************************************
def img_viewer_examples(images, labels, prediction = None, size=0, greyscale=False):
    """
    Visualize output images with thier corrosponding predicted labels and true labels
    \t[Required] images: images batch,
    \t[Required] labels: labels batch,
    \t[optional] prediction: prediction batch to compare the labels with predictions, Default: None
    \t[optional] size: How many image you want to review, Default: images batch size
    \t[optional] greyscale: Set the images to greyscale, Default: False
    """
    batchSize = min(size, images.shape[0])
    
    if size == 0:
        batchSize = images.shape[0]

    # I CAN TAKE THE BATCH_SIZE from the images size/shape according the sent data type
    no_of_columns = round(math.sqrt(batchSize))
    no_of_rows = math.ceil(batchSize / no_of_columns)
    print("batch size {}, no_of_rows {}, no_of_columns {}".format(batchSize, no_of_rows, no_of_columns))
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
        if prediction:
            ax.set_title("{} ({})".format(str(prediction[idx].item()), str(labels[idx].item())),
                    color=("green" if prediction[idx] == labels[idx] else "red"))
        else:
            ax.set_title(str(labels[idx].item()))

def live_graph(trainingLoss, validationLoss = None):
    """
    Draw a graph for training and validation loss\n
    [Required] trainingLoss: Training loss array,\t
    [optional] validationLoss: Validation loss array,
    """
    style.use('fivethirtyeight')
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    def animate(i):
        ax1.cla()
        if y1 is not None:
            ax1.plot(np.array(range(len(y))) + 1, y1, label="Validation loss")
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(len(y), y[-1], y1[-1]))
        else:
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(len(y), y[-1]))
        ax1.plot(np.array(range(len(y))) + 1, y, label="Training loss")
        plt.legend(loc='upper right')
    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.tight_layout()
    plt.show()















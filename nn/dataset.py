import pandas as pd 
import pickle as cPickle
import numpy as np
import numpy
import math 

def unpickle(file):
    
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo,encoding='bytes')
    return dict


class Dataset():

    def __init__(self,file):
        self.file=file
        images= pd.read_csv(file)
        global data
        data=images.values

        """if number of samples in a dataset is odd
        #if number of samples is odd repeat a sample, to handle equal number of records inside a batch
        print(len(data))
        if (len(data)%2 != 0):
            data = numpy.vstack([data, data[0]])
        """

        self.samples=data
        #features
        self.x= data[:, 1:].transpose()
        #label
        self.label= data[:,[0]].transpose()


    def __getitem__(self,index):
        return self.x[ : ,index ], self.label[ : ,index]

    def num_samples(self):
        return len(self.x)

    def get_batch(self, batch_size, batch_iterator):
        #iterating on dataset by batch size=batch_size
        #everytime  get_batch is called it returns different number of samples
        self.x= data[batch_iterator*batch_size : batch_size*(batch_iterator+1) , 1:]
        self.label= data[batch_iterator*batch_size : batch_size*(batch_iterator+1) ,[0]]
        features=numpy.asarray(self.x)
        features=features.transpose()
        labels=numpy.asarray(self.label)
        labels=labels.transpose()
        return features, labels


    def split_data(self,ratio):
        #if ratio =0.6 we multiply it by the whole number of samples
        ratio= int(ratio* self.num_samples())
        train= Dataset(self.file)
        train.x=data[:ratio, 1:]
        train.label=data[:ratio ,[0]]
        train.samples= data[:ratio , :]

        test= Dataset(self.file)
        test.samples= data[ratio: , :]
        test.x=data[ratio:, 1:]
        test.label=data[ratio: ,[0]]
        return train,test



class Data_Loader():
    def __init__(self, dataset,batch_size):
        features =[]
        label=[]
        no_batches = math.ceil(dataset.x.shape[1]/batch_size)

        j = dataset.x.transpose()
        l = dataset.label.transpose()

        s= numpy.asarray(np.array_split(j,int(no_batches)))
        b = numpy.asarray(np.array_split(l,int(no_batches)))

        for j in range(len(s)):
            features.append(s[j].transpose())
        for z in range(len(s)):
            label.append(b[z].transpose())

        self.x=features
        self.label = label


    def __getitem__(self,index):
        return self.x[index], self.label[index]


"""
CIFR-10
"""
#x=unpickle("data_batch_1")
#print(x, "**************************")

"""
MNIST 
"""
# Datasett= Dataset('train.csv')
# #all the labels
# # print(Datasett.x)

# #first sample
# # first_data= Datasett[1]
# # features,label= first_data
# # print(label)


# for i in range(5):
#     f, l = Datasett[i]
#     print(f.shape, l.shape)

# #print(len(features))
# #print(len(Datasett.x))

# #label of first sample
# #print(label)

# #splitting data to train and test 
# train_dataset, test_dataset = Datasett.split_data(0.5)


# #iterating on dataset by batch size=4
# #everytime  get_batch is called it return different four samples
# #for i in range(3):
# #    print(Datasett.get_batch(4,i))


# #Dataloader class
# dataloader=Data_Loader(Datasett,3)
# for x,y in dataloader:
#     print(y.shape)
# my_iter = iter(dataloader.x)
# print(next(my_iter))
# print(next(my_iter))
# print(next(my_iter))
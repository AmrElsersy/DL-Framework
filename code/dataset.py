import pandas as pd 
import pickle as cPickle
import numpy as np
import numpy
import math 
import random


# for loading CIFER-10 dataset
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo,encoding='bytes')
    return dict

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]

class Dataset():

    def __init__(self, x, labels):
        #features
        self.x= x
        #label
        self.label= labels


    def __getitem__(self,index):
        return self.x[ : ,index ], self.label[ : ,index]

    def num_samples(self):
        pixels, samples= self.label.shape
        return samples

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
        train_features= self.x[:,:ratio]
        test_features = self.x[:,ratio:]

        train_labels = self.label[:,:ratio]
        test_labels = self.label[:,ratio :]
        train= Dataset(train_features,train_labels)
        #train.x=data[:ratio, 1:]
        #train.label=data[:ratio ,[0]]
        #train.samples= data[:ratio , :]

        test= Dataset(test_features,test_labels)
        #test.samples= data[ratio: , :]
        #test.x=data[ratio:, 1:]
        #test.label=data[ratio: ,[0]]
        return train,test



class Data_Loader():
    def __init__(self, dataset,batch_size, shuffle=0):
        features =[]
        label=[]
        no_batches = math.ceil(dataset.x.shape[1]/batch_size)


        j = dataset.x.transpose()
        l = dataset.label.transpose()

        if (shuffle == 1):
            randomize = np.arange(len(l))
            np.random.shuffle(randomize)
            l = l[randomize]
            j = j[randomize]

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


class MNIST_dataset(Dataset):

    def __init__(self,file):
        self.file=file
        images= pd.read_csv(file)
        global data
        data=images.values
        self.samples=data
        #features
        self.x= data[:, 1:].transpose()
        #label
        self.label= data[:,[0]].transpose()

        Dataset.__init__(self,self.x,self.label)


class CIFER_10_dataset(Dataset):

    def __init__(self,data_dir,train_flag=1):
        
        labels= []
        feature=None
        l=[]
        if (train_flag ==1 ):
            for i in range(1):
                cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
                if i == 1:
                    feature=cifar_train_data_dict[b'data']
                else:
                    feature=np.vstack((feature, cifar_train_data_dict[b'data']))
         
                labels.extend(cifar_train_data_dict[b'labels'])

        else: #test data
            cifar_test_data_dict = unpickle(data_dir + "/test_batch")
            feature = cifar_test_data_dict[b'data'] 
            labels= cifar_test_data_dict[b'labels']

        l.append(labels)
        feature=feature.reshape(len(feature),3,32,32)
        self.label = numpy.asarray(l)
        self.x= feature.transpose()

        Dataset.__init__(self,self.x,self.label)



"""
CIFR-10
"""
# Datasett= CIFER_10_dataset('cifar-10-batches-py',train_flag=1)
# print("NO of samples:", Datasett.num_samples())
# #print(Datasett.x[0])
# #r, t = Datasett.split_data(0.5)
# #print("r labels", r.label.shape)
# dataloader=Data_Loader(Datasett,1000)
# for x,y in dataloader:
#      print(x.shape)

# """
# MNIST 
# """
#d= MNIST_dataset('train1.csv')

#dataloader2=Data_Loader(d,2, shuffle=1)
#for x,y in dataloader2:
#     print(x)
#     print(y)

import pandas as pd 

class Dataset():

    def __init__(self,file):
        images= pd.read_csv(file)
        data=images.values
        #features
        self.x= data[:, 1:]
        #label
        self.label= data[:,[0]]


    def __getitem__(self,index):
        return self.x[index], self.label[index]



Datasett= Dataset('train.csv')
#all the labels
print(Datasett.label)

#first sample
first_data= Datasett[0]
features,label= first_data

#label of first sample
print(label)
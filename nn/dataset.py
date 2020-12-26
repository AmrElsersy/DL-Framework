import pandas as pd 

class Dataset():

    def __init__(self,file):
        images= pd.read_csv(file)
        global data
        data=images.values
        #features
        self.x= data[:, 1:]
        #label
        self.label= data[:,[0]]


    def __getitem__(self,index):
        return self.x[index], self.label[index]

    def get_batch(self, batch_size, batch_iterator):
        #iterating on dataset by batch size=batch_size
        #everytime  get_batch is called it returns different number of samples
        self.x= data[batch_iterator*batch_size : batch_size*(batch_iterator+1) , 1:]
        self.label= data[batch_iterator*batch_size : batch_size*(batch_iterator+1) ,[0]]
        return self.x, self.label



Datasett= Dataset('train.csv')
#all the labels
print(Datasett.label)

#first sample
first_data= Datasett[0]
features,label= first_data

#label of first sample
print(label)

#iterating on dataset by batch size=4
#everytime  get_batch is called it return different four samples
for i in range(3):
    print(Datasett.get_batch(4,i))
    print("####")
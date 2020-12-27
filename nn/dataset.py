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


class Data_Loader():
    def __init__(self, dataset,batch_size):
        global batch_it
        batch_it=0
        self.d= dataset
        features= []
        labels =[]
        
        for i in range(len(dataset.x)):
            features.append(data[ batch_it*batch_size : batch_size*( batch_it+1) , 1:])
            labels.append(data[batch_it*batch_size : batch_size*(batch_it+1) ,[0]])
            batch_it+=1
        self.d.x=features
        self.d.label=labels

    def __getitem__(self,index):
        return self.d.x[index], self.d.label[index]



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


#Dataloader class
dataloader=Data_Loader(Datasett,4)
my_iter = iter(dataloader)
print(next(my_iter))
print("***")
print(next(my_iter))
import numpy as np



##########################################################
############ Adding some utils functions #################

def initializeFilter(size, scale = 1.0):
        stddev = scale/np.sqrt(np.prod(size))
        return np.random.normal(loc = 0, scale = stddev, size = size)

def initializeBias(size):
        return np.zeros(size)

def initializeWeight(size):
    return np.random.standard_normal(size=size) * 0.01

#########################################################



class forwardPropagationCNN:
    def __init__(self):
        pass
        

    

    def Conv(self, input, filter, bias, stride=1):
        '''
        This function is used to convolves filter over an input image with stride s
        '''
        
        (no_f, no_ch_f, f_s, _) = filter.shape # f_s-> filter size, no_f-> no. of filters, no_ch_f-> no. of channels of filters
        #The filter input is initialized using a standard normal distribution
        filter = initializeFilter(f_s)

        no_ch, in_dim, _ = input.shape # no_ch-> no of channels, in_dim-> input image dimensions

        out_dim = int((in_dim - f_s)/stride)+1

        #error handling, ensuring that the dimension of the image's no of channels matches filter's no of channels
        assert no_ch == no_ch_f, "Dimensions of filter must match dimensions of input image"

        # iitializing output matrix of convolution result
        out = np.zeros((no_f,out_dim,out_dim))

        for specific_f in range(no_f):
            vertical = out_vertical = 0
        # move filter vertically across the image
        while vertical + f_s <= in_dim:
            horizontal = out_horizontal = 0
            # move filter horizontally across the image 
            while horizontal + f_s <= in_dim:
                # perform the convolution operation and add the bias
                out[specific_f, out_vertical, out_horizontal] = np.sum(filter[specific_f] * input[:,vertical:vertical + f_s, horizontal:horizontal + f_s]) + bias[specific_f]
                horizontal += stride
                out_horizontal += 1
            vertical += stride
            out_vertical += 1
        
        return out

    def maxpool(self,image, f_s=2, stride=2):
        '''
        Downsample input `image` using a kernel size of `f` and a stride of `s`
        '''
    
        no_ch, h_prev, w_prev = image.shape
    
        # calculate output dimensions after the maxpooling operation.
        h = int((h_prev - f_s)/stride)+1 
        w = int((w_prev - f_s)/stride)+1
        
        # create a matrix to hold the values of the maxpooling operation.
        downsampled = np.zeros((no_ch, h, w)) 
        
        # slide the window over every part of the image using stride s. Take the maximum value at each step.
        for i in range(no_ch):
            vertical = out_vertical = 0
            # slide the max pooling window vertically across the image
            while vertical + f_s <= h_prev:
                horizontal = out_horizontal = 0
                # slide the max pooling window horizontally across the image
                while horizontal + f_s <= w_prev:
                    # choose the maximum value within the window at each step and store it to the output matrix
                    downsampled[i, out_y, out_x] = np.max(image[i, vertical:vertical+f_s, horizontal:horizontal+f_s])
                    horizontal += stride
                    out_horizontal += 1
                vertical += stride
                out_vertical += 1
        return downsampled

    def make_FC(self, pooled_mat):
        '''
        This functions flatten the pooled layer to change it to a normal feature vector
        to be an input to a multi layer perceptron
        '''
        no_f, dim, dim = pooled_mat.shape
        FC = pooled_mat.reshape(no_f * dim * dim, 1) #flatten the pooled layer
        return FC
    
    def softmax(self, pred):
        """
        This function applies softmax activation function to the raw predictions
        """
        output = np.exp(pred)
        return output/np.sum(output)

    def categoricalCrossEntropy(self, pred, label):
        '''
        calculate the categorical cross-entropy loss of the predictions
        '''
        return -np.sum(label * np.log(pred)) # Multiply the desired output label by the log of the prediction, then sum all values in the vector

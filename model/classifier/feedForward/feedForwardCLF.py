from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.nn import Module
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torch.optim import Adam

SIZE=56
#input_dimension = 3*SIZE*SIZE
hidden_layer_nodes = 10
output_dimension = 2

class FeedforwardClassifier():
    
    def _init_(self):
        pass
    
    
    def trainModel(self):

        class ffNet(nn.Module):
        
            def __init__(self):
                super(ffNet, self).__init__()
                # Linear function
                self.fc0 = nn.Linear(3*SIZE*SIZE, 10)
                #self.fc1 = nn.Linear(input_dimension, hidden_layer_nodes)
                self.fc1 = nn.Linear(10, output_dimension)
                # By default, assume ReLU, so change it later.
                self.activation = nn.ReLU()
                #self.activation = torch.tanh
        
            def forward(self,input):
                """
                The forward prediction path as a feed-forward
                :return:
                :param p: dropout rate
                :param inputs: the regression inputs
                :return: the output prediction
                """
                print('In forward', input.size())
                # input1 = F.interpolate(input,size=(3,SIZE,SIZE), mode='bilinear')
                # input1=input1.view(input1.size(0),3*SIZE*SIZE*10)
                x = self.activation(self.fc0(input))
                print('In forward 2')
                print(x.shape)
                x = F.softmax(self.fc1(x))
                print(x.shape)
                return x
        
        train_path = '/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/data/images/train'
        test_path = '/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/data/images/test'

        device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

       
        # Transforms
        
        transformer=transforms.Compose([
            transforms.Resize((SIZE,SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # To change the pixal range from 0-255 50 0-1, Also changes the datatype to tensors from numpy
            transforms.Normalize([0.8,0.8,0.8], # 
                                 [0.8,0.8,0.8])
            ])
        

        # Create Data loader
        # set path for the train and test datasets 
        train_path = '/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/data/images/train'
        test_path = '/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/data/images/test'
       
        train_loader=DataLoader(
            torchvision.datasets.ImageFolder(train_path,transform=transformer),
            batch_size=3, shuffle= True
            )
        
        test_loader=DataLoader(
            torchvision.datasets.ImageFolder(test_path,transform=transformer),
            batch_size=3, shuffle= True
            )
        
        
        n_iters = 10
        num_epochs=10
        #input_dimension = SIZE*SIZE
        hidden_layer_nodes = 10
        output_dimension = 2
        
        print('Debug 2')
        
        conv = ffNet()
        model=conv.to(device)
        
        criterion = nn.CrossEntropyLoss()

        learning_rate = 0.1
        
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  
        #optimizer=Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
        
        print('Debug 3')
        
        iter = 0
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                # Load images with gradient accumulation capabilities
                #images = images.view(-1,SIZE*SIZE).requires_grad_()
                # Clear gradients w.r.t. parameters
                images = images.view(3,-1)
                print('Printing Image ' , images.size())
                optimizer.zero_grad()
                print('After zero_grad')
                # Forward pass to get output/logits
                outputs = model(images)
                print('output shape : ',outputs.shape)
                # Calculate Loss: softmax --> cross entropy loss
                loss = criterion(outputs, labels)
        
               # Getting gradients w.r.t. parameters
                loss.backward()
        
                # Updating parameters
                optimizer.step()
        
                iter += 1
        
                if iter % 500 == 0:
                    # Calculate Accuracy         
                    correct = 0
                    total = 0
                    # Iterate through test dataset
                    for images, labels in test_loader:
                        #print('Image Shape 2: ',images[0].shape)
                        # Load images with gradient accumulation capabilities
                        #images = images.view(-1, SIZE*SIZE).requires_grad_()
                        images = images.view(1,-1)
                        
                        # Forward pass only to get logits/output
                        outputs = model(images)
        
                        # Get predictions from the maximum value
                        _, predicted = torch.max(outputs.data, 1)
        
                        # Total number of labels
                        total += labels.size(0)
        
                        # Total correct predictions
                        correct += (predicted == labels).sum()
        
                    accuracy = 100 * correct / total
        
                    # Print Loss
                    print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))


#o=FeedforwardClassifier()
#model=o.trainModel()
import os 
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.optim.sgd as optim
from torch.autograd import Variable
import torchvision
import pathlib
from torchvision.models import squeezenet1_1
import torch.functional as F
from io import open
from PIL import Image
import cv2
import streamlit as st
#from p1s3.model.classifier.base.baseClassifier import FeedForwardNet


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


size=150
input_size = 3*150*150 # 784 # 28x28


class feedForwardClassifier():

    def __init__(self):
        pass
    
    def trainModel(self):
        hidden_size = 10 
        num_classes = 2
        num_epochs = 50
        batch_size = 10
        learning_rate = 0.001

        # Transform
        transformer=transforms.Compose([
            transforms.Resize((size,size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # To change the pixal range from 0-255 50 0-1, Also changes the datatype to tensors from numpy
            transforms.Normalize([0.5,0.5,0.5], # 
                                 [0.5,0.5,0.5])
            ])
        
        
        # Create Data loader
        
        train_path = '/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/data/images/train'
        test_path = '/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/data/images/test'
        
        train_loader=DataLoader(
            torchvision.datasets.ImageFolder(train_path,transform=transformer),
            batch_size=batch_size, shuffle= True
            )
        
        test_loader=DataLoader(
            torchvision.datasets.ImageFolder(test_path,transform=transformer),
            batch_size=batch_size, shuffle= False
            )
        
        
        # Fully connected neural network with one hidden layer
        class NeuralNet(nn.Module):
            def __init__(self, input_size, hidden_size, num_classes):
                super(NeuralNet, self).__init__()
                self.input_size = input_size
                self.l1 = nn.Linear(input_size, hidden_size) 
                self.relu = nn.ReLU()
                self.l2 = nn.Linear(hidden_size, num_classes)  
            
            def forward(self, x):
                out = self.l1(x)
                out = self.relu(out)
                out = self.l2(out)
                # no activation and no softmax at the end
                return out
        
        model = NeuralNet(input_size, hidden_size, num_classes).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  
                
        
        # Train the model
        n_total_steps = len(train_loader)
        best_accuracy=0.0
        
        train_count=len(glob.glob(train_path+'/**/*.jpg'))
        test_count=len(glob.glob(test_path+'/**/*.jpg'))
        print(train_count, test_count)
        best_accuracy=0.0
                
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):  
                train_accuracy=0.0
                train_loss=0.0
                # origin shape: [100, 1, 28, 28]
                # resized: [100, 784]
                #images = images.reshape(batch_size,size*size).to(device)
                #print('Image Size : ', images.size())
                labels = labels.to(device)
                
                # Forward pass
                images = images.view(batch_size, -1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss+=loss.cpu().data*images.size(0)
                _,prediction=torch.max(outputs.data,1)
                        
                train_accuracy+=int(torch.sum(prediction==labels.data))
                        
                train_accuracy=train_accuracy/train_count
                train_loss=train_loss/train_count
                if (i+1) % 100 == 0:
                    print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
        
            model.eval()
            test_accuracy=0.0
                        
            with torch.no_grad():
                n_correct = 0
                n_samples = 0
                for images, labels in test_loader:
                    #images = images.reshape(-1, 28*28).to(device)
                    labels = labels.to(device)
                    images = images.view(batch_size, -1)
                    outputs = model(images)
                    # max returns (value ,index)
                    _, predicted = torch.max(outputs.data, 1)
                    n_samples += labels.size(0)
                    n_correct += (predicted == labels).sum().item()
            
                    test_accuracy+=int(torch.sum(prediction==labels.data))
                            
                    test_accuracy=test_accuracy/train_count
                        
                    print('Epoch: '+str(epoch)+' Train Loss: '+str(train_loss)+' Train Accuracy: '+str(train_accuracy)+' Test Accuracy: '+str(test_accuracy))
                    st.write('Epoch: '+str(epoch)+' Train Loss: '+str(train_loss)+' Train Accuracy: '+str(train_accuracy)+' Test Accuracy: '+str(test_accuracy))
            
                    print('test_accuracy',test_accuracy)
                    print('best_accuracy',best_accuracy)
                    if test_accuracy>best_accuracy:
                        torch.save(model.state_dict(),'best_check_point_ffn_model')
                        best_accuracy=test_accuracy
                        st.write('Model Saved with Accuracy : ', best_accuracy)
        
            acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the 10000 test images: {acc} %')


    def infer(self):
        hidden_size = 10 
        num_classes = 2
        batch_size = 1
        
        train_path = '/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/data/images/train'
        predict_path = '/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/data/images/predict'
    
        #Fully connected neural network with one hidden layer
        class NeuralNet(nn.Module):
            def __init__(self, input_size, hidden_size, num_classes):
                super(NeuralNet, self).__init__()
                self.input_size = input_size
                self.l1 = nn.Linear(input_size, hidden_size) 
                self.relu = nn.ReLU()
                self.l2 = nn.Linear(hidden_size, num_classes)  
            
            def forward(self, x):
                out = self.l1(x)
                out = self.relu(out)
                out = self.l2(out)
                # no activation and no softmax at the end
                return out
            
        # categories
        root=pathlib.Path(train_path)
        classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])
        print(classes)
        classes.remove('.DS_Store')
        print(classes)

        checkpoint=torch.load('best_check_point_ffn_model')
        model = NeuralNet(input_size, hidden_size, num_classes).to(device)
        model.load_state_dict(checkpoint)
        model.eval()
        
        # Transforms
        transformer=transforms.Compose([
            transforms.Resize((size,size)),
            transforms.ToTensor(),  # To change the pixal range from 0-255 50 0-1, Also changes the datatype to tensors from numpy
            transforms.Normalize([0.5,0.5,0.5], # 
                                [0.5,0.5,0.5])
        ])                    
        
        # prediction
        def prediction(images_path, transformer):
            print('In Prediction')
            image=Image.open(images_path)
            #images = image.view(batch_size, -1)

            image_tensor=transformer(image).float()
            
            image_tensor=image_tensor.unsqueeze_(0)
            image_tensor=image_tensor.reshape(-1,3*size*size).to(device)
            
            if torch.cuda.is_available():
                image_tensor.cuda()
                
            #input=Variable(image_tensor)
            
            output=model(image_tensor)
            index=output.data.numpy().argmax()
            pred=classes[index]
            
            return pred
        
        images_path=glob.glob(predict_path+'/*.jpg')
        print(images_path)
        
        pred_dict={}

        for i in images_path:
            pred_dict[i[i.rfind('/')+1:]]=prediction(i,transformer)
        
        print(pred_dict)
        return pred_dict
    
    
    
    
    
#o=feedForwardClassifier()
#model=o.trainModel()
#result = o.infer()
#print('Result : ', result)
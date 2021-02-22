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
from p1s3.model.classifier.base.baseClassifier import ConvNet

class cnnClassifier():
    def _init_(self):
        pass
 
    
    def trainModel(self):
                
        # Check for cuda
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        
        # Transforms
        transformer=transforms.Compose([
            transforms.Resize((150,150)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # To change the pixal range from 0-255 50 0-1, Also changes the datatype to tensors from numpy
            transforms.Normalize([0.5,0.5,0.5], # 
                                 [0.5,0.5,0.5])
            ])
        

        # Create Data loader
        # set path for the train and test datasets 
        #train_path = '/Users/spusegao/Documents/DeepLearningWorkshop/projects/imageDetection/images/train'
        #test_path = '/Users/spusegao/Documents/DeepLearningWorkshop/projects/imageDetection/images/validate'
        
        train_path = '/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/data/images/train'
        test_path = '/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/data/images/test'
        
        train_loader=DataLoader(
            torchvision.datasets.ImageFolder(train_path,transform=transformer),
            batch_size=256, shuffle= True
            )
        
        test_loader=DataLoader(
            torchvision.datasets.ImageFolder(test_path,transform=transformer),
            batch_size=256, shuffle= True
            )

        # Data Categories
        root=pathlib.Path(train_path)
        classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])
        print(classes)
        classes.remove('.DS_Store')
        print(classes)
        conv=ConvNet()
        model=conv.to(device)
        
        # Optimizer and Loss Function
        optimizer=Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
        loss_function=nn.CrossEntropyLoss()
        num_epochs=40        

        train_count=len(glob.glob(train_path+'/**/*.jpg'))
        test_count=len(glob.glob(test_path+'/**/*.jpg'))
        print(train_count, test_count)
        
        best_accuracy=0.0
        
        for epoch in range(num_epochs):
            # Evaluation and Training on training data set
            model.train()
            train_accuracy=0.0
            train_loss=0.0
            
            for i, (images,labels) in enumerate(train_loader):
                if torch.cuda.is_available():
                    images=Variable(images.cuda())
                    labels=Variable(labels.cuda())
        
                optimizer.zero_grad()
                outputs=model(images)
                loss=loss_function(outputs,labels)
                loss.backward()
                optimizer.step()
                
                train_loss+=loss.cpu().data*images.size(0)
                _,prediction=torch.max(outputs.data,1)
                
                train_accuracy+=int(torch.sum(prediction==labels.data))
                
            train_accuracy=train_accuracy/train_count
            train_loss=train_loss/train_count
            
            # Evaluation on testing data set
            model.eval()
            test_accuracy=0.0
            
            for i, (image_labels) in enumerate(test_loader):
                if torch.cuda.is_available():
                    images=Variable(images.cuda())
                    labels=Variable(labels.cuda())
                
                outputs=model(images)
                _,prediction=torch.max(outputs.data,1)
                test_accuracy+=int(torch.sum(prediction==labels.data))
                
                test_accuracy=test_accuracy/train_count
            
            #print('Epoch: '+str(epoch)+' Train Loss: '+str(int(train_loss))+'Train Accuracy: '+str(int(train_accuracy))++'Train Accuracy: '+str(int(train_accuracy)))
            print('Epoch: '+str(epoch)+' Train Loss: '+str(train_loss)+' Train Accuracy: '+str(train_accuracy)+' Test Accuracy: '+str(test_accuracy))
            st.write('Epoch: '+str(epoch)+' Train Loss: '+str(train_loss)+' Train Accuracy: '+str(train_accuracy)+' Test Accuracy: '+str(test_accuracy))
         
            #save the model 
            if test_accuracy>best_accuracy:
                torch.save(model.state_dict(),'best_check_point_cnn_model')
                best_accuracy=test_accuracy
                st.write('Model Saved with Accuracy : ', best_accuracy)
        return model



    def infer(self):
        train_path = '/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/data/images/train'
        predict_path = '/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/data/images/predict'

        # categories
        root=pathlib.Path(train_path)
        classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])
        print(classes)
        classes.remove('.DS_Store')
        print(classes)

        checkpoint=torch.load('best_check_point_cnn_model')
        model=ConvNet()
        model.load_state_dict(checkpoint)
        model.eval()
        
        # Transforms
        transformer=transforms.Compose([
            transforms.Resize((150,150)),
            transforms.ToTensor(),  # To change the pixal range from 0-255 50 0-1, Also changes the datatype to tensors from numpy
            transforms.Normalize([0.5,0.5,0.5], # 
                                [0.5,0.5,0.5])
        ])        
        
        # prediction
        def prediction(images_path, transformer):
            
            image=Image.open(images_path)
            image_tensor=transformer(image).float()
            
            image_tensor=image_tensor.unsqueeze_(0)
            
            if torch.cuda.is_available():
                image_tensor.cuda()
                
            input=Variable(image_tensor)
            
            output=model(input)
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


#o=cnnClassifier()
#model=o.trainModel()
#result = o.infer()

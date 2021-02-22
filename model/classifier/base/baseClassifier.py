import glob
import numpy as np 
import cv2
from p1s3.model.classifier.base import baseClassifier
import os 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skimage import io, color
from sklearn.linear_model import LogisticRegression
import torch.nn as nn

class ClassifierBase():
    def __init__(self):
        pass
    
    def loadImageDataForCLF(self, trainDir, testDir):
        #train_dir = "/Users/spusegao/Documents/DeepLearningWorkshop/projects/imageDetection/images/train/*"
        #test_dir = "/Users/spusegao/Documents/DeepLearningWorkshop/projects/imageDetection/images/validate/*"

        train_dir = "/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/data/images/train/*"
        test_dir = "/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/data//images/test/*"

        SIZE = 128 # resizeing every image to size 128 since the deep learning 
           # does not work well with larger images

        images = []  # variabele to store list of training images

        for directory_path in glob.glob(train_dir):
            label_dir = directory_path.split("/")[-1]
            if label_dir=="willow":
                label=0
                #print(label_dir)
                #print(label)
            elif label_dir=="pepper":
                label=1
                #print(label_dir)
                #print(label)
            for img_path in glob.glob(os.path.join(directory_path,"*.jpg")):
                # Reading different files from the folders and corresponding labels (using the folder name)
                #print(img_path)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (SIZE, SIZE))
                img = np.array(img).flatten()
                images.append([img,label])

        for directory_path in glob.glob(test_dir):           
        # for directory_path in glob.glob("/Users/spusegao/Documents/DeepLearningWorkshop/projects/imageDetection/images/validate/*"):
        # the directory path is used to generate a label.
        # all the weeping willow images are stored in a willow directory and all the pepper images in pepper
        # willow and pepper are the labels
        
            label_dir = directory_path.split("/")[-1]
            if label_dir=="willow":
                label=0
                #print(label_dir)
                #print(label)
            elif label_dir=="pepper":
                label=1
                #print(label_dir)
                #print(label) 
            
            for img_path in glob.glob(os.path.join(directory_path,"*.jpg")):
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (SIZE, SIZE))
                img = np.array(img).flatten()       
                images.append([img,0])

        #print("Images")
        #print(len(images))        
        return images

# CNN Class
class ConvNet(nn.Module):
    def __init__(self):
        num_classes=2
        super(ConvNet,self).__init__()
            
        #input shape=(256,3,150,150)
        # (w-f+2P)/s + 1
           
        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        # Image Shape = 256,12,150,150
        self.bn1=nn.BatchNorm2d(num_features=12)
        # Image Shape = 256,12,150,150
        self.relu1=nn.ReLU()
        # Image Shape = 256,12,150,150
            
            
        self.pool=nn.MaxPool2d(kernel_size=2)
        # Image Shape = 256,12,75,75
            
        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        # Image Shape = 256,20,75,75
        self.relu2=nn.ReLU()
        # Image Shape = 256,20,75,75
            
            
            
        self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        # Image Shape = 256,32,75,75
        self.bn3=nn.BatchNorm2d(num_features=32)
        # Image Shape = 256,32,75,75
        self.relu3=nn.ReLU()
        # Image Shape = 256,32,75,75
            
        self.fc=nn.Linear(in_features=32*75*75,out_features=num_classes)
            
    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)
            
        output=self.pool(output)
            
        output=self.conv2(output)
        output=self.relu2(output)
            
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)
            
        # The output above will be in matrix form (256,32,75,75)
        output=output.view(-1,32*75*75)
        output=self.fc(output)
        return output

class FeedForwardNet(nn.Module):
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
 
    # def saveModel(self, modelFileName, model):
    #     pkl_file = open(modelFileName,'wb')
    #     pickle.dump(model,pkl_file)
        
        
    # def loadSavedModel(self, modelFileName):
    #     pkl_file = open(modelFileName,'rb')
    #     pkl_model = pickle.load(pkl_file)
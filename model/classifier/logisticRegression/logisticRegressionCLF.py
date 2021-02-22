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
from sklearn.linear_model import LogisticRegression
from skimage import io, color
from p1s3.model.classifier.base.baseClassifier import ClassifierBase
import streamlit as st

SIZE = 128

# Create Data loader
# set path for the train and test datasets
#train_dir = '/Users/spusegao/Documents/DeepLearningWorkshop/projects/imageDetection/images/train'
#test_dir = '/Users/spusegao/Documents/DeepLearningWorkshop/projects/imageDetection/images/validate'        
train_dir = '/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/data/images/train'
test_dir = '/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/data/images/test'

#b = baseClassifier
#images=[]
#images = BaseClassifier.loadImageDataForCLF(trainDir=train_dir, testDir=test_dir)
#print(len(images))

class logisticRegressionClassifier(ClassifierBase):
    def __init__(self):
        pass
    
    def trainModel(self):
        print('In train model')
        images=[]
        
        cfb = ClassifierBase()
        images = cfb.loadImageDataForCLF(trainDir=train_dir, testDir=test_dir)
        print('In train model - post loadImageDataForCLF')

        random.shuffle(images)
        features = []
        labels =[]
        
        for feature ,label in images:
            #feature = feature.reshape(feature.shape[0], -1)
            #print(feature.shape)
            features.append([feature])
            labels.append([label])
            
        
        X_train, X_test, y_train, y_test = train_test_split(features,labels, test_size=0.25)
        
        
        X_train = np.array(X_train)
        nsamples, nx, ny = X_train.shape
        X_train = X_train.reshape((nsamples,nx*ny))
        X_train.shape
        
        model = LogisticRegression()
        model.fit(X_train, y_train)    
        print('Saving Model')
        pkl_file = open('LR_ImageClassifier','wb')
        print('Saving Model - Calling Dump')
        pickle.dump(model,pkl_file)
        print('Done Saving Model')

        X_test = np.array(X_test)
        ntsamples, ntx, nty = X_test.shape
        X_test = X_test.reshape((ntsamples,ntx*nty))
        X_test.shape
        
        prediction=model.predict(X_test)        
        accuracy = model.score(X_test, y_test)
        print('Accuracy: ', accuracy)
        st.write('Accuracy : ' , accuracy)
        return model
    
    # def saveModel(self, in_model):
    #     pkl_file = open('LR_ImageClassifier','wb')
    #     pickle.dump(in_model,pkl_file)
    #     #ClassifierBase.saveModel(modelFileName='RF_ImageClassifier',model=model)
    #     pass    
        
        
    def predict(self):
        SIZE = 128
        # Loading saved model to predict
        print("In predict")
        pkl_file = open('LR_ImageClassifier','rb')
        savedModel = pickle.load(pkl_file)
        print("In predict - model loaded")

        # Read image from a pre-determined folder
        #img_path='/Users/spusegao/Documents/DeepLearningWorkshop/projects/imageDetection/images/test/test.jpg'
        img_path='/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/data/images/predict/test.jpg'
        
        images=[]
        img = cv2.imread(img_path,cv2.IMREAD_COLOR)
        #img = cv2.resize(SIZE,SIZE)
        img = cv2.resize(img,(SIZE,SIZE))
        img = np.array(img).flatten()
        label=2
        images.append([img,label])
        features=[]
        labels=[]
        for feature, label in images:
            features.append([feature])
            labels.append([label])


        X_test = features
        X_test = np.array(X_test)
        #print(X_test)
        #print('Image Length')
        #print(len(X_test))
        nsamples, nx, ny = X_test.shape
        X_test = X_test.reshape((nsamples,nx*ny))
        X_test.shape
        
        
        prediction = savedModel.predict(X_test)
        print('prediction is', prediction[0])
        result=prediction[0]
        print('Result is : ' , result)
        return result
    
o=logisticRegressionClassifier()
#o.trainModel()
model=o.predict()
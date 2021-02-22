import glob
import numpy as np 
import cv2
from p1s3.model.classifier.base import baseClassifier
import os 
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skimage import io, color
import streamlit as st
import pandas as pd
from p1s3.model.classifier.base.baseClassifier import ClassifierBase
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from yellowbrick.classifier import confusion_matrix, roc_auc, class_prediction_error, classification_report
import skimage.io
import skimage.segmentation
import copy
import skimage.viewer

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

class randomForestClassifier(ClassifierBase):
    def __init__(self):
        pass
    
    
    def trainModel(self):
        images=[]
        
        cfb = ClassifierBase()
        images = cfb.loadImageDataForCLF(trainDir=train_dir, testDir=test_dir)
        print('In RF train model - post loadImageDataForCLF')

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
        print('X_train shape')
        print(X_train[10].shape)
        #print('X_train[1] Length')
        #print(len(X_train[1])
        nsamples, nx, ny = X_train.shape
        X_train = X_train.reshape((nsamples,nx*ny))
        X_train.shape
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)    
        
        
        X_test = np.array(X_test)
        ntsamples, ntx, nty = X_test.shape
        X_test = X_test.reshape((ntsamples,ntx*nty))
        X_test.shape
        
        
        prediction=model.predict(X_test)        
        accuracy = model.score(X_test, y_test)
        print('Accuracy: ', accuracy)
        st.write('Accuracy : ' , accuracy)
        
        
        print(prediction[3])
        plt.imshow(X_test[3].reshape(128,128,3))
        print("pepper => 0 ; willow => 1")   
        
        pkl_file = open('RF_ImageClassifier','wb')
        pickle.dump(model,pkl_file)
        
        # confmatrix = confusion_matrix(y_test, labels)
        # print(confmatrix)
        # #st.write(confmatrix)

        # Confusion Matrix
        
        confusion_matrix(model, X_train, y_train, X_test, y_test, classes=[0,1],
                         title='Confusion matrix for the Random Forest Classifier')

        # Classification Report 
        classification_report(model, X_train, y_train, X_test, y_test, classes=['0.0','1.0'], support=True,
                      title='Classification report visualized as a heat map')
        
        # ROC Curve
        roc_auc(model, X_train, y_train, X_test=X_test, y_test=y_test, classes=['0.0','1.0'],
        title='ROC Curve using Random Forest')
        
        # Prediction Error Plot
        #class_prediction_error(model, X_train, y_train, X_test, y_test, classes=['0.0','1.0'],
        #              title='Class Prediction Error Report for Random Forest', );
        
        
        # report=classification_report(y_test,labels)
        # print(report)
       
        
        #return model
    
    def saveModel(self, in_model):
        pkl_file = open('RF_ImageClassifier','wb')
        pickle.dump(model,pkl_file)
        #ClassifierBase.saveModel(modelFileName='RF_ImageClassifier',model=model)
        pass    



    def explain(self):

        def perturb_image(img, perturbation,segments):
            active_pixels = np.where(perturbation == 1)[0]
            mask = np.zeros(segments.shape)
            
            for active in active_pixels:
                mask[segments == active] = 1
                
            perturbed_image = copy.deepcopy(img)
            perturbed_image = perturbed_image*mask[:,:,np.newaxis]
            return perturbed_image
        

        print('In explain')

        print('Before displaying the image')
        
                     
        # Create perturbastions of image : 
        # Extract superpixels from image 
        img_path='/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/data/images/predict/test.jpg'
        
        img = skimage.io.imread('/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/data/images/predict/test.jpg')
        skimage.io.imshow(img)
        img = skimage.transform.resize(img,(299,299))

        print('Explain - Image Length',len(img))

        superpixels = skimage.segmentation.quickshift(img, kernel_size=4,max_dist=200, ratio=0.2)
        num_superpixels = np.unique(superpixels).shape[0]
        print(num_superpixels)
        
        skimage.io.imshow(skimage.segmentation.mark_boundaries(img/2+0.5, superpixels))
        
        # Create random perturbations
        
        num_perturb = 150
        perturbations = np.random.binomial(1,0.5, size=(num_perturb, num_superpixels))
        print(perturbations[0])
        
        skimage.io.imshow(perturb_image(img/2+0.5,perturbations[0],superpixels))
        
        
        pkl_file = open('RF_ImageClassifier','rb')
        savedModel = pickle.load(pkl_file)
        print('Predicting using saved model')
        
        predictions = []
        for pert in perturbations:
            perturbed_img = perturb_image(img,pert,superpixels)
            ptd = perturbed_img[np.newaxis,:,:,:]
            ptd=np.array(ptd).flatten()
            print('Calling predict from explain',ptd.shape)
            pred = savedModel.predict(ptd)
            predictions.append(pred)
            predictions = np.array(predictions)
            predictions.shape
            
            
            
            
            
            
        
    def predict(self,fileName=""):
        SIZE=128
        # Loading saved model to predict
        print("In predict")
        
        # Read image from a pre-determined folder
        if fileName=="":
            pkl_file = open('RF_ImageClassifier','rb')
            savedModel = pickle.load(pkl_file)
            print("In predict - model loaded")
            print(savedModel)
            img_path='/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/data/images/predict/test.jpg'
            img = cv2.imread(img_path,cv2.IMREAD_COLOR)
            img = cv2.resize(img,(SIZE,SIZE))
            img = np.array(img).flatten()
        else:
            img = fileName


        print('printing Image shape : ' , img.shape)
        
        images=[]    
        label=2
        images.append([img,label])
        features=[]
        labels=[]
        for feature, label in images:
            features.append([feature])
            labels.append([label])

        X_test = features
        X_test = np.array(X_test)
        print(X_test)
        print('Image Length')
        print(len(X_test))
        nsamples, nx, ny = X_test.shape
        X_test = X_test.reshape((nsamples,nx*ny))
        X_test.shape
        
        prediction = savedModel.predict(X_test)
        print('prediction is', prediction[0])
        result=prediction[0]
        print('Result is : ' , result)
        #print('Result Accuracy is : ', accuracy)
        
        #randomForestClassifier.explain(self,image_path=img_path,model=savedModel)
        return result

    
    def compareModels(self):
        
        images=[]
        
        cfb = ClassifierBase()
        images = cfb.loadImageDataForCLF(trainDir=train_dir, testDir=test_dir)
        print('In RF train model - post loadImageDataForCLF')

        random.shuffle(images)
        features = []
        labels =[]
        
        for feature ,label in images:

            features.append([feature])
            labels.append([label])
            
        
        X_train, X_test, y_train, y_test = train_test_split(features,labels, test_size=0.25)
        
        
        X_train = np.array(X_train)
        nsamples, nx, ny = X_train.shape
        X_train = X_train.reshape((nsamples,nx*ny))
        X_train.shape
        print('Setting params')
        
        model_params = {
            'svm':{
                'model': svm.SVC(gamma='auto'),
                'params': {
                    'C':[1],
                    'kernel':['rbf','linear']
                    }
                },
            'random forest':{
                'model':RandomForestClassifier(),
                'params': {
                    'n_estimators':[10,20,30,40,50]
                    }
                },
            'logistic Regression':{
                'model':LogisticRegression(solver='liblinear',multi_class='auto'),
                'params':{
                    'C':[1]
                    }
                }
            }
        
        scores=[]
        print('Done Setting params')
        for model_name, mp in model_params.items():
            grid = GridSearchCV(estimator=mp['model'],
                                param_grid=mp['params'],
                                cv=5,
                                n_jobs=16,
                                return_train_score=False)

            print('Done Setting grid')
            
            grid.fit(X_train,y_train)

            print('Done grid.fit')
            scores.append({'model':model_name,
                           'best_score':grid.best_score_,
                           'best_params':grid.best_params_})
            
        df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
        
        print(df)
        st.write(df)
        
        return df


        
#myobject = randomForestClassifier()
#myobject.compareModels()

#print("Training-----------------------------------")
#model = myobject.trainModel()   
#print("Predicting-----------------------------------")
#print("pepper => 1 ; willow => 0")
#myobject.predict()
#myobject.explain()   
#myobject.saveModel(in_model=model)

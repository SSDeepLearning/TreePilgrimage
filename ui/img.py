import streamlit as st
import urllib
from PIL import Image
from p1s3.model.classifier.randomForest.randomForestCLF import randomForestClassifier
from p1s3.model.classifier.logisticRegression.logisticRegressionCLF import logisticRegressionClassifier
from p1s3.model.classifier.cnn.cnnCLF import cnnClassifier
from p1s3.model.classifier.pretrained.preTrained_ResNet import preTrainedResnetClassifier
from p1s3.model.classifier.feedForward.ffnCLF import feedForwardClassifier
import requests



in_operation=st.selectbox("Select Operation",("Analyze Image","Train Model","Compare Models"))

if in_operation =="Compare Models":
    clf = randomForestClassifier()
    result = clf.compareModels() 
    in_image_file=None
    in_text_input=None

if in_operation=="Train Model":
    in_model=st.selectbox("Select Model to Train",("","Logistic Regression","Pre-trained Model","Feed Forward Network","Convolutional Network","Random Forest"))
    in_image_file=None
    in_text_input=None

if in_operation=="Analyze Image":
    in_model=st.selectbox("Select Model to Analyze Image",("","Logistic Regression","Pre-trained Model","Feed Forward Network","Convolutional Network","Random Forest"))
    in_text_input=st.text_area(label='Enter image URL here ...', value="Hello", height=1, max_chars=1000)
    in_image_file=None

    # Store the file locally
if in_text_input is not None:
    if in_text_input=="":
        pass
    elif in_text_input=='Hello':
        pass
    else:
        url=in_text_input
        myfile=requests.get(url)
        st.write(myfile)
        open('../data/images/predict/test.jpg','wb').write(myfile.content)
        #st.image('../data/images/predict/test.jpg')


if in_operation =="Train Model":
    if in_model == "Logistic Regression": 
        print('Debugging')
        clf = logisticRegressionClassifier()
        model = clf.trainModel()
    if in_model == "Random Forest": 
        print('Debugging')
        clf = randomForestClassifier()
        model = clf.trainModel()
    if in_model == "Convolutional Network": 
        print('Debugging')
        clf = cnnClassifier()
        model = clf.trainModel()
    if in_model == "Feed Forward Network": 
        print('Debugging')
        clf = feefForwardClassifier()
        model = clf.trainModel()

if in_operation =="Analyze Image":
    if in_model == "Logistic Regression":
        print('LR Prediction')
        clf = logisticRegressionClassifier()
        result = clf.predict()
        if result==0:
            result='willow'
        elif result==1:
            result='pepper'    
        st.write('Prediction is : ', result)    
    if in_model=="Random Forest":
        print('RF Prediction')
        clf = randomForestClassifier()
        result = clf.predict()
        print(result)
        if result==0:
            result='willow'
        elif result==1:
            result='pepper'    
        st.write('Prediction is : ', result)    
    if in_model=="Convolutional Network":
        print('CNN Prediction')
        clf = cnnClassifier()
        result = clf.infer()
        print(result)
        if result==0:
            result='willow'
        elif result==1:
            result='pepper'    
        st.write('Prediction is : ', result)        
    if in_model=="Pre-trained Model":
        print('Pre-trained Prediction')
        clf = preTrainedResnetClassifier()
        result = clf.infer()
        print(result)
        #if result==0:
        #    result='willow'
        #elif result==1:
        #    result='pepper'    
        st.write('Prediction is : ', result)  
    if in_model=="Feed Forward Network":
        print('Feed Forward Network')
        clf = feedForwardClassifier()
        result = clf.infer()
        print(result)
        #if result==0:
        #    result='willow'
        #elif result==1:
        #    result='pepper'    
        st.write('Prediction is : ', result)  
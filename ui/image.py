import streamlit as st
import urllib
from PIL import Image
from p1s3.model.classifier.randomForest.randomForestCLF import randomForestClassifier
from p1s3.model.classifier.logisticRegression.logisticRegressionCLF import logisticRegressionClassifier
from p1s3.model.classifier.cnn.cnnCLF import cnnClassifier
from p1s3.model.classifier.pretrained.preTrained_ResNet import preTrainedResnetClassifier
from p1s3.model.classifier.feedForward.ffnCLF import feedForwardClassifier
import requests



in_operation=st.sidebar.radio("Select Operation",("Analyze Image","Train Model","Compare Models","Explain Predictions", "Neural Style Transfer"))

if in_operation =="Compare Models":
    clf = randomForestClassifier()
    result = clf.compareModels() 
    in_image_file=None
    in_text_input=None

if in_operation=="Train Model":
    in_model = st.sidebar.radio("Select Model to Train",("","Logistic Regression","Pre-trained Model","Feed Forward Network","Convolutional Network","Random Forest"))

#    in_model=st.selectbox("Select Model to Train",("","Logistic Regression","Pre-trained Model","Feed Forward Network","Convolutional Network","Random Forest"))
    in_image_file=None
    in_text_input=None

if in_operation=="Analyze Image":
    in_model=st.sidebar.radio("Select Model to Analyze Image",("","Logistic Regression","Pre-trained Model","Feed Forward Network","Convolutional Network","Random Forest"))
    in_text_input=st.text_area(label='Enter image URL here ...', value="Hello", height=1, max_chars=1000)
    in_image_file=None

if in_operation=="Neural Style Transfer":
    #if in_style == "View Pre-styled":
    in_preselected = st.sidebar.radio("View precreated",("","Willow + Metamorphosis","Red Pepper + Taj Mahal"))
    if in_preselected == "Willow + Metamorphosis":
        images = ['/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/model/WillowInMetamorphosisStyle.png',
        '/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/model/transfer/willow.jpg',
        '/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/model/transfer/Metamorphosis.jpg']
        st.markdown('**  Weeping Willow In Salvador Dali (Metamorphosis) Style **')
        st.image(images,width=210)

    if in_preselected == "Red Pepper + Taj Mahal":
        images = ['/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/model/RedPepperInTajmahalStyle.png',
        '/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/model/transfer/PeruvianRedPepperTree.jpg',
        '/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/model/transfer/TajMahal.jpg']
        st.markdown('**  Peruvian Red Pepper In Taj Mahal Style **')
        st.image(images,width=210)

    in_style = st.sidebar.radio("Transfer your own : Select Style",("","Taj Mahal"))
    if in_style =="Taj Mahal":
        in_text_input=st.text_area(label='Enter image URL here ...', value="Hello", height=1, max_chars=1000)
        in_style_input=st.text_area(label='Enter style image URL here ...', value="Hello", height=1, max_chars=1000)
    in_text_input=""

if in_operation=="Explain Predictions":
    in_text_input=""
    in_explain_predictions=st.sidebar.radio("View Prior Prediction Explanations",("","Red Pepper Prediction","Weeping Willow Prediction"))
    if in_explain_predictions == "Red Pepper Prediction":
        st.markdown('**  Why Red Pepper? **')
        w_images = ["/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/ui/redPepperPrediction/r0.png",
        "/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/ui/redPepperPrediction/r1.png",
        "/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/ui/redPepperPrediction/r2.png",
        "/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/ui/redPepperPrediction/r3.png",
        "/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/ui/redPepperPrediction/r4.png",
        "/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/ui/redPepperPrediction/r5.png"
        ]
        st.image(w_images,width=210)
        #st.image("/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/ui/redPepperPrediction/explainPepperUsingLime.png")
        #st.image("/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/ui/redPepperPrediction/OriginalRedPepper.jpg")

    elif in_explain_predictions == "Weeping Willow Prediction":
        st.markdown('**  Why Weeping Willow? **')
        w_images = ['/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/ui/weepingWillowPredictions/w0.png',
        '/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/ui/weepingWillowPredictions/w2.png',
        '/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/ui/weepingWillowPredictions/w3.png',
        '/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/ui/weepingWillowPredictions/w4.png',
        '/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/ui/weepingWillowPredictions/w5.png']
        st.image(w_images,width=210)

    in_predict_your_own = st.sidebar.radio("Predict and Explain",("","Using resnext101"))

    if in_predict_your_own == "Using resnext101":
        in_text_input=st.text_area(label='Enter image URL here ...', value="Hello", height=1, max_chars=1000)

    in_image_file=None

    # Store the file locally
if in_text_input is not None:
    if in_text_input=="":
        pass
    elif in_text_input=="Hello":
        pass
    else:
        url=in_text_input
        myfile=requests.get(url)
        st.write(myfile)
        open('../data/images/predict/test.jpg','wb').write(myfile.content)
        st.image(url)


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
from __future__ import print_function, division

import os
import io
import requests 
import random
import time
import numpy as np
import copy
import cv2
import pdb
import shap
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import itertools

import torchvision.datasets as dset
import pathlib
import glob
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from skimage.segmentation import mark_boundaries
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
from lime import lime_image

import streamlit as st


plt.ion()   # interactive mode
random.seed(42)

# data dir
DATA_DIR = os.path.join("/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/data", "images")

# model parameters
MODEL_NAME = "resnext101"
NUM_CLASSES = 2
FIXED_FEATURE_EXTRACTOR = True # True --> Base model weights fixed

# input size of image
INPUT_SIZE = (460, 460)
#INPUT_SIZE = (150, 150)

# data parameters
BATCH_SIZE = 30
SHUFFLE = True
NUM_WORKERS = 0

# optimizer hyperparameters
LEARNING_RATE = 0.001
USE_ADAM_OPTIM = True

# SGD optimizer hyperparameters
MOMENTUM = 0.9

# Adam optimizer hyperparameters
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-08
WEIGHT_DECAY = 1e-5 #0.0

# training hyperparameters
NUM_EPOCHS = 25

# lr learning rate scheduler hyperparameters
DECAY_STEP_SIZE = 5
GAMMA = 0.1

# cyclic learning rate scheduler hyperparameters
BASE_LR = 0.001
MAX_LR = 0.005
STEP_SIZE_UP = 2000
STEP_SIZE_DOWN = None
MODE_CYCLIC = "triangular" # "triangular2" or "exp_range"

# ReduceLROnPlateau learning rate scheduler hyperparameters
MODE_PLATEAU = "min" # "max"
FACTOR = 0.1
PATIENCE = 2
COOLDOWN = 0
MIN_LR = 0

# other
RANDOM = 42


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

# Create Data loader
# set path for the train and test datasets

train_dir = '/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/data/images/train'
test_dir = '/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/data/images/test'

def initialize_model(model_name, num_classes, feature_extract, custom_input_size, use_pretrained=True):
    print('Model Name passed 1 : ', model_name)

    model = None
    input_size = 0
    filnalconv_name =  ""
    
    if model_name =="resnet152":
        model = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = (224, 224)
        finalconv_name = "layer4"  
    elif model_name == "inception":
        model = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        # Handle the auxilary net
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs,num_classes)
        input_size = (299, 299)
        finalconv_name = "Mixed_7c"   
    
    elif model_name == "resnext101":
        """ ResNext101
        """
        print('Model Name passed 2 : ', model_name)
        model = models.resnext101_32x8d(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = (224, 224)
        #input_size = (150,150)

        finalconv_name = "layer4"
        
    else:
        print("Invalid model name")
        exit()    
    
    params_to_update = set_params_to_update(model,feature_extract)
    
    if (not INPUT_SIZE[0] is None) and (not INPUT_SIZE[1] is None):
        input_size = custom_input_size
        
    return model, input_size, params_to_update, finalconv_name

def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
            
def set_params_to_update(model, feature_extract):
    params_to_update=[]
    if feature_extract:
        for param in model.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
    else:
        params_to_update = model.parameters()
    return params_to_update

net, input_size, params_to_update, finalconv_name = initialize_model(MODEL_NAME, NUM_CLASSES, 
                                                                     FIXED_FEATURE_EXTRACTOR, 
                                                                     INPUT_SIZE)
                
net = net.to(device)

print(net)

class RandomDiscreteRotation(object):
    def __init__(self, angles):
        self.discrete_angles= angles
        
    def __call__(self, img):
        angle = random.choice(self.discrete_angles)
        return transforms.functional.rotate(img,angle)
    
# define data transformations
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize(input_size),
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        RandomDiscreteRotation([0, 90, 180, 270]), # TODO --> depending on input size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
        
# create datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                  for x in ["train", "test"]}

image_datasets

#data = {
#    'train':
#    datasets.ImageFolder(root=train_dir, transform=data_transforms['train']),
#    'test':
#    datasets.ImageFolder(root=test_dir, transform=data_transforms['test'])
#    }
#Dataloader iterators, make sure to shuffle
#dataloaders = {
#    'train': torch.utils.data.DataLoader(data['train'], batch_size=BATCH_SIZE, shuffle=True,num_workers=0),
#    'test': torch.utils.data.DataLoader(data['test'], batch_size=BATCH_SIZE, shuffle=True,num_workers=0)
#}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS)
              for x in ["train","test"]}
dataloaders

dataset_sizes = {x: len(x) for x in ['train', 'test']}
print(dataset_sizes)


class_names = image_datasets['train'].classes
print(class_names)

def imshow(images, title=None):
    images = images.numpy().transpose(1,2,0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    images = std * images + mean
    images = np.clip(images, 0, 1)
    plt.figure(figsize=(16,10))
    plt.imshow(images)
    if title is not None:
        plt.title(title)
    #plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()

# Get a batch of training data
inputs, classes = next(iter(dataloaders["train"]))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

def trainModel(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, is_inception=False):
    since = time.time()
    
    # list for tracking
    val_acc_history = []
    val_loss_history = []
    train_acc_history = []
    train_loss_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs -1))
        print("-" * 10)
        
        for phase in ["train", "test"]:
            if phase=="train":
                # scheduler.step()
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over the data
            
            for inputs, labels in (dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == "train"):
                    if is_inception and phase == "train":
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds==labels.data)
                
            epoch_loss= running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print("{}Loss : {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            
            # deep copy the model
            
            if phase =="test" and train_acc_history[-1] > 0.9 and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase =="train":
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
            if phase == "test":
                scheduler.step(epoch_loss)
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
                
        print()
    
    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(best_acc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, val_loss_history, train_acc_history, train_loss_history
                
# define optimizer
if USE_ADAM_OPTIM:
    optimizer = optim.Adam(params_to_update, lr=LEARNING_RATE, betas=(BETA_1, BETA_2), eps=EPSILON, weight_decay=WEIGHT_DECAY)
else:
    optimizer = optim.SGD(params_to_update, lr=LEARNING_RATE, momentum=MOMENTUM)

# define objective
if NUM_CLASSES <= 2:
    objective = nn.CrossEntropyLoss() #nn.BCELoss()
else:
    objective = nn.CrossEntropyLoss()

# define learning rate scheduler
#lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=DECAY_STEP_SIZE, gamma=GAMMA)
#lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=BASE_LR, max_lr=MAX_LR, step_size_up=STEP_SIZE_UP, step_size_down=STEP_SIZE_DOWN, mode=MODE_CYCLIC)
lr_scheduler =  optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=MODE_PLATEAU, factor=FACTOR, patience=PATIENCE, verbose=False, cooldown=COOLDOWN, min_lr=MIN_LR)     


# train model
net, val_acc_history, val_loss_history, train_acc_history, train_loss_history = trainModel(net, dataloaders, objective, optimizer, lr_scheduler, num_epochs=NUM_EPOCHS, is_inception=(MODEL_NAME=="inception"))                

# Evaluate model on the test data                    
net.eval()

def get_test_results(model, dataloader):
    label = np.array([])
    label_predicted = np.array([])
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        
        _,preds = torch.max(outputs, 1)
        
        # append labels
        label = np.append(label, labels.numpy())
        label_predicted = np.append(label_predicted, preds.cpu().numpy())
        
    return label, label_predicted

label, label_predicted = get_test_results(net, dataloaders["test"])

len(label), len(label_predicted)

label_list = label.tolist()
label_predicted_list = label_predicted.tolist()

# create a classification report
print(classification_report(label_list, label_predicted_list, target_names=class_names))


# save and load state dict
#save_path = "inceptionV3_dict"
MODEL_DIR = os.path.join("/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/model/classifier/pretrained")
SAVE_PATH = os.path.join(MODEL_DIR,"models",MODEL_NAME + ".pt")
torch.save(net.state_dict(),SAVE_PATH)
 
# load model
MODEL_NAME = "resnext101.pt"
LOAD_PATH = os.path.join(MODEL_DIR,"models",MODEL_NAME)


#net, input_size, params_to_update, finalconv_name = initialize_model(MODEL_NAME=="resnext101", NUM_CLASSES,FIXED_FEATURE_EXTRACTOR, INPUT_SIZE)
## Initialize Model Alternate code 
model = models.resnext101_32x8d(pretrained=True)
set_parameter_requires_grad(model, FIXED_FEATURE_EXTRACTOR)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
input_size = (224, 224)
#input_size = (150, 150)
finalconv_name = "layer4"

net.load_state_dict(torch.load('/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/model/classifier/pretrained/models/resnext101.pt'))
net.eval()

save_path = os.path.join("models",MODEL_NAME + ".pt")
torch.save(net,SAVE_PATH)

LOAD_PATH = os.path.join(MODEL_DIR,"models",MODEL_NAME)

# load Model

net = torch.load(LOAD_PATH)
net.eval()

def image_to_tensor_and_normalize(img):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return preprocess(img)

def read_image_and_transform(path,model_input_size=input_size,tensor_and_normalize=False):
    img = Image.open(path)
    img
    preprocess=transforms.Compose([
        transforms.Resize(model_input_size)
    ])
    transform = preprocess(img)
    
    if tensor_and_normalize: 
        transform=image_to_tensor_and_normalize(transform)
    return transform


def batch_predict(images, model=net):
    model.eval()

    batch = torch.stack(tuple(image_to_tensor_and_normalize(i) for i in images), dim=0)
    
    model = model.to(device)
    batch = batch.to(device)
    
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

img = Image.open('/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/data/images/predict/test.jpg')

#data_dir = '/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/data/images/'
img_path = os.path.join(DATA_DIR,'predict','test.jpg')

test_pred = batch_predict([read_image_and_transform(img_path,input_size, False)])
idx = test_pred.squeeze().argmax()

print("Predicted Class :  {}".format(class_names[idx]))

explainer = lime_image.LimeImageExplainer(random_state=RANDOM)
explanation = explainer.explain_instance(np.array(read_image_and_transform(img_path,input_size,False)),
                                         batch_predict,
                                         labels=np.array([0]),
                                         num_features=2,
                                         num_samples=100,
                                         random_seed=RANDOM)

plt.imshow(explanation.segments)

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
img_boundry1 = mark_boundaries(temp/255.0, mask)
plt.imshow(img_boundry1)

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=1, hide_rest=True)
img_boundry2 = mark_boundaries(temp/255.0, mask)
plt.imshow(img_boundry2)

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=True)
img_boundry3 = mark_boundaries(temp/255.0, mask)
plt.imshow(img_boundry3)

gigapixels = mark_boundaries(np.zeros((explanation.image.shape)), explanation.segments)
test_img = 0.2 * (explanation.image/255.0) + 0.7 *  img_boundry2 + 0.1 * gigapixels
plt.imshow(test_img)




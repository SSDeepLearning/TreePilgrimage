from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy
import torchvision.datasets as dset
import pathlib
import glob
from io import open
from PIL import Image
import cv2
import streamlit as st
from torch.autograd import Variable
import skimage.io
import skimage.segmentation
import copy
import skimage.viewer
import tensorflow as tf

SIZE = 128

# Create Data loader
# set path for the train and test datasets

train_dir = '/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/data/images/train'
test_dir = '/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/data/images/test'


class preTrainedResnetClassifier():


    def trainModel(self):
        num_epochs=10
        
        
        # Image transformations
        image_transforms = {
            # Train uses data augmentation
            'train':
            transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224), # Image net standards
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]) # Imagenet standards
            ]),
                # Test does not use augmentation
            'test':
            transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ]),
        }
           
        batch_size = 4
           
        data = {
            'train':
            datasets.ImageFolder(root=train_dir, transform=image_transforms['train']),
            'test':
            datasets.ImageFolder(root=test_dir, transform=image_transforms['test'])
            }
        # Dataloader iterators, make sure to shuffle
        dataloaders = {
            'train': torch.utils.data.DataLoader(data['train'], batch_size=batch_size, shuffle=True,num_workers=0),
            'test': torch.utils.data.DataLoader(data['test'], batch_size=batch_size, shuffle=True,num_workers=0)
        }
        
        
        dataset_sizes = {x: len(data[x]) for x in ['train', 'test']}
        print(dataset_sizes)
        
        
        class_names = data['train'].classes
        print(class_names)
        
        
        device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
        
        
        model = models.resnet34(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        
       
        criterion = nn.CrossEntropyLoss()
        # Observe that all parameters are being optimized
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        
        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    
        since = time.time()
    
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
    
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
    
            # Each epoch has a training and validation phase
            for phase in ['train', 'test']:
                if phase == 'train':
                    scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode
    
                current_loss = 0.0
                current_corrects = 0
    
                # Here's where the training happens
                print('Iterating through data...')
    
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
    
                    # We need to zero the gradients, don't forget it
                    optimizer.zero_grad()
    
                    # Time to carry out the forward training poss
                    # We only need to log the loss stats if we are in training phase
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
    
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
    
                    # We want variables to hold the loss statistics
                    current_loss += loss.item() * inputs.size(0)
                    current_corrects += torch.sum(preds == labels.data)
    
                epoch_loss = current_loss / dataset_sizes[phase]
                epoch_acc = current_corrects.double() / dataset_sizes[phase]
    
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
    
                # Make a copy of the model if the accuracy on the validation set has improved
                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
    
            print()
    
        time_since = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_since // 60, time_since % 60))
        print('Best test Acc: {:4f}'.format(best_acc))
    
        # Now we'll load in the best model weights and return it
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(),'best_check_point_pretrained_model')
        
        return model    
    
    def infer(self):
    
        train_path = '/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/data/images/train'
        predict_path = '/Users/spusegao/Documents/DeepLearningWorkshop/p1s3/data/images/predict/'

        # categories
        root=pathlib.Path(train_path)
        classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])
        print(classes)
        classes.remove('.DS_Store')
        print(classes)

        checkpoint=torch.load('best_check_point_pretrained_model')
        model=models.resnet34(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model.load_state_dict(checkpoint)
        model.eval()
        
        # Transforms    
    
        transformer=transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224), # Image net standards
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]) # Imagenet standards
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



    def explain(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        def perturb_image(img, perturbation,segments):
            active_pixels = np.where(perturbation == 1)[0]
            mask = np.zeros(segments.shape)
            
            for active in active_pixels:
                mask[segments == active] = 1
                
            perturbed_image = copy.deepcopy(img)
            perturbed_image = perturbed_image*mask[:,:,np.newaxis]
            return perturbed_image
        
        def prediction(images_path,transformer):
            
            print('In prediction')
            image = images_path
            #image=Image.open(images_path)            
            #image_tensor=image.float()            
            #image_tensor=image_tensor.unsqueeze_(0)            
            #if torch.cuda.is_available():
            #    image_tensor.cuda()
            #    print('debug 4')
            
            image_tensor =torch.tensor(image, device=device).float()
            #image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)    
            input=Variable(image_tensor)
            print('debug 5')
            
            output=model(input)
            print('debug 6')
            index=output.data.numpy().argmax()
            print('debug 7')
            pred=classes[index]
            print('debug 8')
            
            return pred
        
        
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
        print('perturbations',perturbations[0])
        
        skimage.io.imshow(perturb_image(img/2+0.5,perturbations[0],superpixels))
        print('Perturbed Image Size : ', perturb_image(img/2+0.5,perturbations[0],superpixels))
        transformer=transforms.Compose([
            transforms.RandomResizedCrop(size=299, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=288), # Image net standards
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]) # Imagenet standards
        ])
        
        checkpoint=torch.load('best_check_point_pretrained_model')
        model=models.resnet34(pretrained=True)


        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model.load_state_dict(checkpoint)
        model.eval()

        print('saved model loaded')
        
        predictions = []
        for pert in perturbations:
            perturbed_img = perturb_image(img,pert,superpixels)
            ptd = perturbed_img[np.newaxis,:,:,:]
            #ptd=np.array(ptd).flatten()
            print('Calling predict from explain',ptd.shape)
            pred =prediction(ptd,transformer)
        
            print(pred)
            
            #predictions.append(pred)
            #predictions = np.array(predictions)
            #predictions.shape
            
            
 

#o=preTrainedResnetClassifier()
#o.explain()
#model=o.trainModel()
#model=o.infer()
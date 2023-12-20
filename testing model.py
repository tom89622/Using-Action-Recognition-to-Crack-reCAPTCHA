# -*- coding: utf-8 -*- 
from __future__ import print_function, division 
from sklearn.model_selection import train_test_split,cross_val_score 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.optim import lr_scheduler 
from torch.autograd import Variable 
import numpy as np 
import torchvision 
from torchvision import datasets, models, transforms 
import matplotlib.pyplot as plt 
import time 
import os 
import cv2 
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM 
from pytorch_grad_cam.utils.image import show_cam_on_image 
from pytorch_grad_cam import GuidedBackpropReLUModel 
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image,preprocess_image 
 
plt.ion()   # interactive mode 
 
 
data_transforms = {  
    'test': transforms.Compose([ 
        transforms.Resize((480,480)), 
        # transforms.CenterCrop( ( 240, 240 ) ), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ]) 
} 
 
data_dir = 'OurData' # data location 
batch_size = 1
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), 
                                          data_transforms[x]) 
                  for x in ['test']} 
dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, # batch_size 批量大小將決定我們一次訓練的樣本數目，將影響到模型的優化程度和速度。 
                                             shuffle=False, num_workers=0) # num worker  
              for x in ['test']} 
dataset_sizes = {x: len(image_datasets[x]) for x in ['test']} 
class_names = image_datasets['test'].classes 
 
use_gpu = torch.cuda.is_available()

model_ft = torch.load( 'train_M_6_level_1.pt' ) 
images, labels = next(iter(dataloders['test'])) 
 
def imshow(inp, title=None): 
    """Imshow for Tensor.""" 
    inp = inp.numpy().transpose((1, 2, 0)) 
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225]) 
    inp = std * inp + mean 
    inp = np.clip(inp, 0, 1) 
    plt.imshow(inp) 
    if title is not None: 
        plt.title(title) 
    plt.pause(0.001)  # pause a bit so that plots are updated 
 
def visualize_model(model, num_images=12): 
    images_so_far = 0 
    fig = plt.figure() 
 
    for i, data in enumerate(dataloders['test']): 
        inputs, labels = data 
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda()) 
        else: 
            inputs, labels = Variable(inputs), Variable(labels) 
 
        outputs = model(inputs) 
        _, preds = torch.max(outputs.data, 1) 
 
        for j in range(inputs.size()[0]): 
            images_so_far += 1 
            ax = plt.subplot(num_images//2, 3, images_so_far) 
            ax.axis('off') 
            ax.set_title('predicted: {}'.format(class_names[preds[j]])) 
            imshow(inputs.cpu().data[j]) 
 
            if images_so_far == num_images: 
                return 
 
 
target_layer = model_ft.layer4[-1] 
num = 201
indexOfAnsList = 0

# prepare to count predictions for each class 
correct_pred = {classname: 0 for classname in class_names} 
total_pred = {classname: 0 for classname in class_names} 

ansList = []
# again no gradients needed 
with torch.no_grad(): 
    index = 1
    for data in dataloders['test']: 
        images, labels = data 
        images, labels = Variable(images.cuda()), Variable(labels.cuda()) 
        model_ft = model_ft.cuda()
        outputs = model_ft(images)
        _, predictions = torch.max(outputs.data, 1)
        print( outputs, end = '' )
        # collect the correct predictions for each class 
        for label, prediction in zip(labels, predictions): 

            print( 'No.', index, prediction )
            index = index + 1 
            ansList.append( prediction.cpu() )
            if label == prediction: 
                correct_pred[class_names[label]] += 1 
                 
            total_pred[class_names[label]] += 1 

# print accuracy for each class 
for classname, correct_count in correct_pred.items(): 
    if ( total_pred[classname] != 0 ): 
        accuracy = 100 * float(correct_count) / total_pred[classname] 
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
    else :
        print( classname, "no data" )
                    

for data in dataloders['test']: 
    input_tensor, label = data 
    # Note: input_tensor can be a batch tensor with several images! 
 
    # Construct the CAM object once, and then re-use it on many images: 
    cam = GradCAM(model=model_ft, target_layer=target_layer, use_cuda=use_gpu) 
 
    # If target_category is None, the highest scoring category 
    # will be used for every image in the batch. 
    # target_category can also be an integer, or a list of different integers 
    # for every image in the batch. 
    target_category = label    
 
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing. 
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category) 
    all_grayscale_cam = grayscale_cam 
     
    for index in range( batch_size ): 
        if index < np.size( all_grayscale_cam, axis = 0 ) : 
            # In this example grayscale_cam has only one image in the batch: 
            grayscale_cam_one = all_grayscale_cam[index, :] 
 
            rgb_img = input_tensor[index, :] 
            rgb_img = rgb_img.numpy().transpose((1, 2, 0)) 
            mean = np.array([0.485, 0.456, 0.406]) 
            std = np.array([0.229, 0.224, 0.225]) 
            rgb_img = std * rgb_img + mean 
            rgb_img = np.clip(rgb_img, 0, 1)
            cam_image = show_cam_on_image(  rgb_img, grayscale_cam_one, use_rgb = True ) 
 
            # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding. 
            cam_image = cv2.cvtColor( cam_image, cv2.COLOR_RGB2BGR ) 
             
            if index > 0 and label[index-1] != label[index] : 
              num = 1
             
            cv2.imwrite(f'./Cam_picture/No.{num}_test_cam_Ans={label[index]}_Predict={ansList[indexOfAnsList]}.jpg', cam_image) 

            num = num + 1 
            indexOfAnsList = indexOfAnsList + 1

 

# visualize_model( model_ft )
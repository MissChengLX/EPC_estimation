# -*- coding: cp936 -*-
from torchvision import models, transforms
import torch.nn as nn
import torch

from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
from torchsummary import summary
from torch.utils.data import DataLoader
import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
use_cuda = torch.cuda.is_available()
device= torch.device("cuda:0" if use_cuda else "cpu")

import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as M

def get_resnet():
    arch = 'resnet50'
    model_file = './resnet50_places365.pth.tar'
    model = models.__dict__[arch](num_classes=365)

    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    state_dict.pop('fc.weight')
    state_dict.pop('fc.bias')
    #print(state_dict)
    model_dict =model.state_dict() 
    model_dict.update(state_dict) 
    model.load_state_dict(model_dict)  

    # frozen the parameters
    for name, value in model.named_parameters():
        print(name)
        if (name.split('.')[0] == 'bn1'):
            value.requires_grad = True
        elif (name.split('.')[0] == 'fc'):
            value.requires_grad = True
        elif (len(name.split('.'))>2) and (name.split('.')[2][:2] =='bn'):
            #print(name.split('.')[2][:2])
            value.requires_grad = True
        elif name =='layer4.2.conv3.weight':
            value.requires_grad =True  # retrained the last convolutional layers
        else:
            value.requires_grad = False

    model.fc = nn.Sequential(
        nn.Flatten(),
        nn.BatchNorm1d(2048),
        nn.Dropout(0.5),
        nn.Linear(2048, 512), #512
        nn.ReLU(),  # activation layer
        nn.Dropout(0.4),
        #nn.BatchNorm1d(512),
        nn.Linear(512, 4),
        nn.LogSoftmax(dim=1),  # lossfuction
    )
    return model



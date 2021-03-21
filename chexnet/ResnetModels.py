import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import torchvision

class ResNet18(nn.Module):

    def __init__(self, classCount, isTrained):
	
        super(ResNet18, self).__init__()

        self.resnet18 = torchvision.models.resnet18(pretrained=isTrained)

        kernelCount = self.resnet18.fc.out_features

        self.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())


    def forward(self, x):
        x = self.resnet18(x)
        x = self.classifier(x)
        return x

class ResNet50(nn.Module):

    def __init__(self, classCount, isTrained):
	
        super(ResNet50, self).__init__()
		
        self.resnet50 = torchvision.models.resnet50(pretrained=isTrained)

        kernelCount = self.resnet50.fc.out_features
		
        self.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.resnet50(x)
        x = self.classifier(x)
        return x
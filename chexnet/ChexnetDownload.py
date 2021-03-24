import os
import numpy as np
import time
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as func

from sklearn.metrics.ranking import roc_auc_score

from DensenetModels import DenseNet121
from DensenetModels import DenseNet169
from DensenetModels import DenseNet201

from ResnetModels import ResNet18
from ResnetModels import ResNet50

from DatasetGenerator import DatasetGenerator


#-------------------------------------------------------------------------------- 
    
def download (nnArchitecture, nnIsTrained, nnClassCount):

    #-------------------- SETTINGS: NETWORK ARCHITECTURE
    if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, nnIsTrained)
    elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained)
    elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained)
    elif nnArchitecture == 'RES-NET-18': model = ResNet18(nnClassCount, nnIsTrained)
    elif nnArchitecture == 'RES-NET-50': model = ResNet50(nnClassCount, nnIsTrained)
    
    model = torch.nn.DataParallel(model)

if __name__ == "__main__":
    download("DENSE-NET-169", True, 14)
    # download("RES-NET-50", True, 14)
                


import os
import numpy as np
import time
import sys

import pdb

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
import torch.cuda as cutorch

from sklearn.metrics.ranking import roc_auc_score

from DensenetModels import DenseNet121
from DensenetModels import DenseNet169
from DensenetModels import DenseNet201

from ResnetModels import ResNet18
from ResnetModels import ResNet50

from DatasetGenerator import DatasetGenerator

from tqdm import tqdm

#--------------------------------------------------------------------------------

class ChexnetTrainer ():

    #---- Train the densenet network
    #---- pathDirData - path to the directory that contains images
    #---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    #---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    #---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    #---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    #---- nnClassCount - number of output classes
    #---- trBatchSize - batch size
    #---- trMaxEpoch - number of epochs
    #---- transResize - size of the image to scale down to (not used in current implementation)
    #---- transCrop - size of the cropped image
    #---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    #---- checkpoint - if not None loads the model and continues training

    def train (pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, transResize, transCrop, launchTimestamp, checkpoint):


        #-------------------- SETTINGS: NETWORK ARCHITECTURE
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'RES-NET-18': model = ResNet18(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'RES-NET-50': model = ResNet50(nnClassCount, nnIsTrained).cuda()

        model = torch.nn.DataParallel(model).cuda()

        #-------------------- SETTINGS: DATA TRANSFORMS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        transformList = []
        transformList.append(transforms.RandomResizedCrop(transCrop))
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        transformSequence=transforms.Compose(transformList)

        #-------------------- SETTINGS: DATASET BUILDERS
        datasetTrain = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTrain, transform=transformSequence)
        datasetVal =   DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileVal, transform=transformSequence)

        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=24, pin_memory=True)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=24, pin_memory=True)

        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam (model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')

        #-------------------- SETTINGS: LOSS
        loss = torch.nn.BCELoss(size_average = True)

        #---- Load checkpoint
        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])


        #---- TRAIN THE NETWORK

        lossMIN = 100000

        for epochID in range (0, trMaxEpoch):

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime

            ChexnetTrainer.epochTrain (model, dataLoaderTrain, optimizer, scheduler, trMaxEpoch, nnClassCount, loss, epochID)
            val_loss = ChexnetTrainer.epochVal (model, dataLoaderVal, optimizer, scheduler, trMaxEpoch, nnClassCount, loss, epochID)

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime

            scheduler.step(val_loss)

            if val_loss < lossMIN:
                lossMIN = val_loss
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, './models/m-' + launchTimestamp + '.pth.tar')
                print ('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(val_loss))
            else:
                print ('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(val_loss))

    #--------------------------------------------------------------------------------

    def epochTrain (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss, epoch):

        model.train()
        losses = AverageMeter()

        iterator = tqdm(enumerate (dataLoader), total=len(dataLoader))
        for batchID, (input, target) in iterator:

            target = target.cuda()
            output = model(input)

            lossvalue = loss(output, target)

            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()
            losses.update(lossvalue.item())
            desc = f'Train: Loss {losses.avg:.4f} | Epoch: {epoch+1}/{epochMax}'
            iterator.set_description(desc)
            iterator.refresh()

            del lossvalue

    #--------------------------------------------------------------------------------

    def epochVal (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss, epoch):

        model.eval ()
        losses = AverageMeter()


        iterator = tqdm(enumerate (dataLoader), total=len(dataLoader))
        for i, (input, target) in iterator:

            target = target.cuda()
            output = model(input)

            losstensor = loss(output, target)

            losses.update(losstensor.item())
            desc = f'Val: Loss {losses.avg:.4f}'
            iterator.set_description(desc)
            iterator.refresh()

        return losses.avg

    #--------------------------------------------------------------------------------

    #---- Computes area under ROC curve
    #---- dataGT - ground truth data
    #---- dataPRED - predicted data
    #---- classCount - number of classes

    def computeAUROC (dataGT, dataPRED, classCount):

        outAUROC = []

        for i in range(classCount):
            outAUROC.append(roc_auc_score(dataGT[:, i], dataPRED[:, i]))

        return outAUROC


    #--------------------------------------------------------------------------------

    #---- Test the trained network
    #---- pathDirData - path to the directory that contains images
    #---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    #---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    #---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    #---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    #---- nnClassCount - number of output classes
    #---- trBatchSize - batch size
    #---- trMaxEpoch - number of epochs
    #---- transResize - size of the image to scale down to (not used in current implementation)
    #---- transCrop - size of the cropped image
    #---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    #---- checkpoint - if not None loads the model and continues training

    def test (pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, transResize, transCrop, launchTimeStamp):


        CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

        cudnn.benchmark = True

        #-------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'RES-NET-18': model = ResNet18(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'RES-NET-50': model = ResNet50(nnClassCount, nnIsTrained).cuda()

        model = torch.nn.DataParallel(model).cuda()

        modelCheckpoint = torch.load(os.path.join("models", pathModel))
        model.load_state_dict(modelCheckpoint['state_dict'])

        #-------------------- SETTINGS: DATA TRANSFORMS, TEN CROPS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        #-------------------- SETTINGS: DATASET BUILDERS
        transformList = []
        # transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.TenCrop(transCrop))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        transformSequence=transforms.Compose(transformList)

        datasetTest = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTest, transform=transformSequence)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, num_workers=8, shuffle=False, pin_memory=True)

        # outGT = torch.FloatTensor().cuda()
        # outPRED = torch.FloatTensor().cuda()

        outGT = []
        outPRED = []

        model.eval()

        for i, (input, target) in enumerate(dataLoaderTest):

            # outGT = torch.cat((outGT, target), 0)
            outGT.append(target.numpy())

            bs, n_crops, c, h, w = input.size()

            out = model(input.view(-1, c, h, w).cuda())
            outMean = out.detach().cpu().numpy().reshape(bs, n_crops, -1).mean(axis=1)

            # outPRED = torch.cat((outPRED, outMean.data), 0)
            outPRED.append(outMean)
            del out

        outGT = np.concatenate(outGT, axis=0)
        outPRED = np.concatenate(outPRED, axis=0)

        aurocIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED, nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()

        modelTested = pathModel.split("/")[-1].replace(".pth.tar", "")
        modelResultsPath = os.path.join("results", modelTested + ".txt")

        with open(modelResultsPath, "w") as f:
            print ('Architecture: ', nnArchitecture, file=f)
            print ('AUROC mean ', aurocMean, file=f)
            for i in range (0, len(aurocIndividual)):
                print (CLASS_NAMES[i], ' ', aurocIndividual[i], file=f)

        print ('Architecture: ', nnArchitecture)
        print ('AUROC mean ', aurocMean)
        for i in range (0, len(aurocIndividual)):
            print (CLASS_NAMES[i], ' ', aurocIndividual[i])


        return
#--------------------------------------------------------------------------------

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count





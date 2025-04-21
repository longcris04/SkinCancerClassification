import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as f
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from torchvision import models

import os
import argparse

def image_transform(image):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=(0, 180)),
        transforms.ColorJitter(brightness=.2, hue=.05)
    ])
    return transform(image)

class CancerDataset(Dataset):
    def __init__(self, xroot, yroot, train, indices):
        self.x = np.load(xroot, mmap_mode='r')
        self.y = np.load(yroot, mmap_mode='r')
        
        self.x = self.x[indices]/255
        self.y = self.y[indices]
        self.x = torch.tensor(self.x, dtype=torch.float)
        
        self.y = torch.tensor(self.y, dtype=torch.long)
        
        if train:
            print("Augmentation Start!")
            x_aug = [torch.stack(list(map(image_transform, self.x))) for _ in range(4)]
            x_aug.append(self.x)
            self.x = torch.cat(x_aug)
            self.y = torch.cat([self.y]*5)
            print("Augmentation Completed! Number of Data: ", self.x.shape[0])
        
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=False, default="resnet50")
    parser.add_argument("--pretrained", required=False, default=False)
    parser.add_argument("--num_class", required=False, default=3, type=int)
    parser.add_argument("--cancer_only", required=True, type=bool)
    
    args = parser.parse_args()  
    
    if args.model_name == 'alexnet':
        model = models.alexnet()
        model.classifier[6] = nn.Linear(4096,args.num_class,bias=True)
    
    if args.model_name == 'resnet18':
        model = models.resnet18()
        model.fc = nn.Linear(512, args.num_class, bias=True)
        
    if args.model_name == 'resnet34':
        model = models.resnet34()
        model.fc = nn.Linear(512, args.num_class, bias=True)

    if args.model_name == 'resnet50':
        if args.pretrained:
            model = models.resnet50(weights="IMAGENET1K_V1")
        model = models.resnet50()
        model.fc = nn.Linear(2048,args.num_class)
        
    if args.model_name == 'resnet101':
        model = models.resnet101()
        model.fc = nn.Linear(2048,args.num_class)
    
    if args.model_name == 'resnet152':
        model = models.resnet152()
        model.fc = nn.Linear(2048,args.num_class)
        
    if args.model_name == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s()
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, args.num_class, bias=True)
        )
    
    if args.model_name == 'efficientnet_v2_m':
        model = models.efficientnet_v2_m()
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, args.num_class, bias=True)
        )
        
    if args.model_name == 'efficientnet_v2_l':
        model = models.efficientnet_v2_l()
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, args.num_class, bias=True)
        )
    
    if args.model_name == 'inception_v3':
        model = models.inception_v3()
        model.fc = nn.Linear(2048,args.num_class,bias=True)
        model.aux_logits = False
        model.AuxLogits = None
    
    if args.model_name == 'vit_b_16':
        model = models.vit_b_16()
        model.heads = nn.Linear(768,args.num_class,bias=True)
    
    if args.model_name == 'vit_b_32':
        model = models.vit_b_32()
        model.heads = nn.Linear(768,args.num_class,bias=True)
        
    if args.model_name == 'vit_l_16':
        model = models.vit_l_16()
        model.heads = nn.Linear(768,args.num_class,bias=True)
        
    if args.model_name == 'vit_l_32':
        model = models.vit_l_32()
        model.heads = nn.Linear(768,args.num_class,bias=True)  
        
    if args.model_name == 'vit_h_14':
        model = models.vit_h_14()
        model.heads = nn.Linear(768,args.num_class,bias=True)
        
    if args.model_name == 'densenet121':
        model = models.densenet121()
        model.fc = nn.Linear(1024,args.num_class)
    
    if args.model_name == 'densenet161':
        model = models.densenet161()
        model.fc = nn.Linear(1024,args.num_class)
    
    if args.model_name == 'mobilenet_v2':
        model = models.mobilenet_v2()
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(1280, args.num_class, bias=True)
        )
        
    if args.model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large()
        model.classifier = nn.Sequential(
            nn.Linear(960,1280,bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2,inplace=True),
            nn.Linear(1280,args.num_class,bias=True)
        )
    
    if args.model_name == 'resnext50_32x4d':
        model = models.resnext50_32x4d()
        model.fc = nn.Linear(2048,args.num_class,bias=True)
        
    if args.model_name == 'resnext101_32x8d':
        model = models.resnext101_32x8d()
        model.fc = nn.Linear(2048,args.num_class,bias=True)
        
    if args.model_name == 'regnet_y_400mf':
        model = models.regnet_y_400mf()
        model.fc = nn.Linear(440,args.num_class,bias=True)
        
    if args.model_name == 'regnet_y_800mf':
        model = models.regnet_y_800mf()
        model.fc = nn.Linear(784,args.num_class,bias=True)
        
    if args.model_name == 'regnet_y_1_6gf':
        model = models.regnet_y_1_6gf()
        model.fc = nn.Linear(888,args.num_class,bias=True)
        
    if args.model_name == 'regnet_y_3_2gf':
        model = models.regnet_y_3_2gf()
        model.fc = nn.Linear(1512,args.num_class,bias=True)
    
    if args.model_name == 'regnet_y_8gf':
        model = models.regnet_y_8gf()
        model.fc = nn.Linear(2016,args.num_class,bias=True)
        
#     if args.model_name in ['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14']:
#         cancer_dataset = CancerDataset('images.npy', 'labels.npy')
#     else:
#         cancer_dataset = CancerDataset('images.npy', 'labels.npy')
    
    generator = torch.Generator().manual_seed(0)
    if args.cancer_only:
        image_link = 'cancer_only_images.npy'
        label_link = 'cancer_only_labels.npy'
    else:
        image_link = 'images.npy'
        label_link = 'labels.npy'
    num_data = np.load(image_link, mmap_mode='r').shape[0]
    train_indices, test_indices = torch.utils.data.random_split(list(range(num_data)), [0.9,0.1], generator=generator)
    train_dataset = CancerDataset(image_link, label_link, train=True, indices=train_indices)
    test_dataset = CancerDataset(image_link, label_link, train=False, indices=test_indices)
#     train_dataset, test_dataset = torch.utils.data.random_split(cancer_dataset, [0.9,0.1], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100)
    
#     class_weight = np.unique(np.load(label_link, mmap_mode='r')[train_indices], return_counts=True)[1]
#     class_weight = 1-class_weight/class_weight.sum()
#     class_weight = torch.tensor(class_weight, dtype=torch.float)

    device = 'cuda'
    
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)

    E = 100
    best_acc = 0
    for e in range(E):
        print('Epoch: ', e)
        accuracy = 0
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            total_loss += loss.item()*x.shape[0]
            loss.backward()
            optimizer.step()
            accuracy += (torch.max(output, dim=1).indices==y).sum().item()
        
        accuracy /= len(train_dataset)
        total_loss /= len(train_dataset)
        scheduler.step(total_loss)
        print("Train Loss: ",total_loss, "\tTrain Accuracy: ", accuracy)
        
        with torch.inference_mode():
            model.eval()
            test_acc = 0
            test_total_loss = 0
            for x, y in tqdm(test_loader):
                x, y = x.to(device), y.to(device)
                test_output = model(x)
                test_loss = loss_fn(test_output, y)
                test_total_loss += test_loss.item()*x.shape[0]
                test_acc += (torch.max(test_output, dim=1).indices==y).sum().item()
            test_acc /= len(test_dataset)
            test_total_loss /= len(test_dataset)
            print("Test Loss: ",test_total_loss, "\tTest Accuracy: ", test_acc)
            if test_acc >= best_acc:
#             if True:
                torch.save(model.state_dict(), 'best_' + args.model_name + '.pt')
                best_acc = test_acc
                print('New best model found!')
    print("Best model performance: ", best_acc)

if __name__ == "__main__":
    main()
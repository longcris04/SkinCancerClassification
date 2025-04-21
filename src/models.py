
from torchvision import models
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, mobilenet_v2, MobileNet_V2_Weights
# from torchinfo import summary


def CNNModel(model_name,num_class=3,image_size=None):
    


    if model_name == 'alexnet':
        model = models.alexnet()
        model.classifier[6] = nn.Linear(4096,num_class,bias=True)
    
    if model_name == 'resnet18':
        model = models.resnet18()
        model.fc = nn.Linear(512, num_class, bias=True)
        
    if model_name == 'resnet34':
        model = models.resnet34()
        model.fc = nn.Linear(512, num_class, bias=True)

    if model_name == 'resnet50':
        
        # model = models.resnet50(weights="IMAGENET1K_V1")
        model = models.resnet50()
        model.fc = nn.Linear(2048,num_class)
        
    if model_name == 'resnet101':
        model = models.resnet101()
        model.fc = nn.Linear(2048,num_class)
    
    if model_name == 'resnet152':
        model = models.resnet152()
        model.fc = nn.Linear(2048,num_class)
        
    if model_name == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s()
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_class, bias=True)
        )
    
    if model_name == 'efficientnet_v2_m':
        model = models.efficientnet_v2_m()
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_class, bias=True)
        )
        
    if model_name == 'efficientnet_v2_l':
        model = models.efficientnet_v2_l()
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_class, bias=True)
        )
    
    if model_name == 'inception_v3':
        model = models.inception_v3()
        model.fc = nn.Linear(2048,num_class,bias=True)
        model.aux_logits = False
        model.AuxLogits = None
    
    if model_name == 'vit_b_16':
        model = models.vit_b_16()
        model.heads = nn.Linear(768,num_class,bias=True)
    
    if model_name == 'vit_b_32':
        model = models.vit_b_32()
        model.heads = nn.Linear(768,num_class,bias=True)
        
    if model_name == 'vit_l_16':
        model = models.vit_l_16()
        model.heads = nn.Linear(768,num_class,bias=True)
        
    if model_name == 'vit_l_32':
        model = models.vit_l_32()
        model.heads = nn.Linear(768,num_class,bias=True)  
        
    if model_name == 'vit_h_14':
        model = models.vit_h_14()
        model.heads = nn.Linear(768,num_class,bias=True)
        
    if model_name == 'densenet121':
        model = models.densenet121()
        model.fc = nn.Linear(1024,num_class)
    
    if model_name == 'densenet161':
        model = models.densenet161()
        model.fc = nn.Linear(1024,num_class)
    
    if model_name == 'mobilenet_v2':
        model = models.mobilenet_v2()
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(1280, num_class, bias=True)
        )
        
    if model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large()
        model.classifier = nn.Sequential(
            nn.Linear(960,1280,bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2,inplace=True),
            nn.Linear(1280,num_class,bias=True)
        )
    
    if model_name == 'resnext50_32x4d':
        model = models.resnext50_32x4d()
        model.fc = nn.Linear(2048,num_class,bias=True)
        
    if model_name == 'resnext101_32x8d':
        model = models.resnext101_32x8d()
        model.fc = nn.Linear(2048,num_class,bias=True)
        
    if model_name == 'regnet_y_400mf':
        model = models.regnet_y_400mf()
        model.fc = nn.Linear(440,num_class,bias=True)
        
    if model_name == 'regnet_y_800mf':
        model = models.regnet_y_800mf()
        model.fc = nn.Linear(784,num_class,bias=True)
        
    if model_name == 'regnet_y_1_6gf':
        model = models.regnet_y_1_6gf()
        model.fc = nn.Linear(888,num_class,bias=True)
        
    if model_name == 'regnet_y_3_2gf':
        model = models.regnet_y_3_2gf()
        model.fc = nn.Linear(1512,num_class,bias=True)
    
    if model_name == 'regnet_y_8gf':
        model = models.regnet_y_8gf()
        model.fc = nn.Linear(2016,num_class,bias=True)





    return model


















if __name__ == '__main__':
    model = CNNModel('efficientnet_v2_l')
    # print(model)
    # summary(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"device used: {device}")
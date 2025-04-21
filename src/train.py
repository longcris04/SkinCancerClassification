import torch
from sklearn.model_selection import StratifiedKFold
from datasets import MyDataset
import os
import numpy as np
import cv2
import gc

from PIL import Image
import shutil
import matplotlib.pyplot as plt 
from torchvision.transforms import Resize, ToTensor, Compose
from torch.utils.data import Dataset, DataLoader
from models import CNNModel
from tqdm.autonotebook import tqdm
import torch.nn as nn
import torch.optim as optim
# from torchvision.models import resnet18, ResNet18_Weights, mobilenet_v2, MobileNet_V2_Weights, resnet50, ResNet50_Weights
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datasets import get_list_img_label
from torch.utils.tensorboard import SummaryWriter



def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="cool")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('Confusion_matrix', figure, epoch)

def train():
    
    lr = 1e-2
    batch_size = 128
    num_fold = 5
    data_path = "./../../Datasets/BVDLTW/data_raw/all/raw"
    tensorboard_dir = "./../log/tensorboard/"
    if os.path.exists(tensorboard_dir):
        shutil.rmtree(tensorboard_dir)
    os.makedirs(tensorboard_dir)
    
    checkpoint_dir = "./../log/checkpoint/"
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir)
    
    
    classes = [44,45,4]
    num_class = len(classes)
    model_names = [
                #    'efficientnet_v2_l',
                   'resnet50',
                   'resnet18',
                   'efficientnet_v2_s',
                   
                   'inception_v3', 
                   'mobilenet_v2', 
                   'alexnet',
                   'densenet121',
                   'mobilenet_v3_large',
                   'resnext50_32x4d',
                   'regnet_y_400mf',
                   'regnet_y_1_6gf',
                   ]
    transform = Compose([
        ToTensor(),
        Resize(size=(224,224))
    ])
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"device used: {device}")
    print("-------------------------")
    
    img_paths, labels_lists = get_list_img_label(data_path,classes)
    
    skf = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=42)
    
    for fold, (train_idx,val_idx) in enumerate(skf.split(img_paths,labels_lists)):
        print(f"Fold {fold}")
        
        fold_tensorboard_dir = os.path.join(tensorboard_dir,f"fold {fold}")
        os.makedirs(fold_tensorboard_dir)
        print(f"successfully create {fold_tensorboard_dir}")
        
        fold_checkpoint_dir = os.path.join(checkpoint_dir,f"fold {fold}")
        os.makedirs(fold_checkpoint_dir)
        print(f"successfully create {fold_checkpoint_dir}")
        
        
            
        train_imgs, val_imgs = img_paths[train_idx], img_paths[val_idx]
        train_labels, val_labels = labels_lists[train_idx], labels_lists[val_idx]
        
        train_dataset = MyDataset(image_paths=train_imgs,labels=train_labels, transform=transform)
        val_dataset = MyDataset(image_paths=val_imgs,labels=val_labels, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        for model_name in model_names:
            
            log = {}
            
            best_acc = -1
            
            
            model_fold_tensorboard_dir = os.path.join(fold_tensorboard_dir, f"{model_name}")
            os.makedirs(model_fold_tensorboard_dir)
            writer = SummaryWriter(log_dir=model_fold_tensorboard_dir)
            
            model_fold_checkpoint_dir = os.path.join(fold_checkpoint_dir, f"{model_name}")
            os.makedirs(model_fold_checkpoint_dir)
            
            print(f"---------------")
            print(f"Start training for model {model_name} in fold {fold}/{num_fold}:")
            
            
            
            
            model = CNNModel(model_name,num_class)
            model.to(device)
            
            optimizer = optim.Adam(params=model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            
            for epoch in range(epochs):
                model.train()
                losses_train = []
                avg_loss_train = -1
                progress_bar = tqdm(train_loader,colour="cyan")
                for iter, (images,labels) in enumerate(progress_bar):
                    # print(image.shape,label)
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(images)
                    
                    loss = criterion(outputs,labels)
                    losses_train.append(loss.item())
                    
                    writer.add_scalar(tag="Train/Loss", scalar_value=loss.item(), global_step=(epoch)*len(train_loader)+iter)
                    progress_bar.set_description(f"Epoch: {epoch}/{epochs}. Loss: {loss:0.4f}")
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # print(f"iter {iter}/{len(train_loader)}: loss: {loss.item()}")
                   
                avg_loss_train = np.mean(losses_train)
                
                print(f"Epoch {epoch}/{epochs}: Loss train : {avg_loss_train}")
                
                progressbar = tqdm(val_loader, colour='yellow')
                model.eval()
                model.eval()

                losses_val = []
                avg_loss_val = -1
                labels_all = []
                predictions_all = []
                
                for iter,(images,labels) in enumerate(progressbar):
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs,labels)
                    writer.add_scalar(tag="Val/Loss", scalar_value=loss.item(), global_step=(epoch)*len(val_loader)+iter)
                    progress_bar.set_description(f"Validation. Loss: {loss:0.4f}")
                    losses_val.append(loss.item())
                    predictions = torch.argmax(outputs,dim=1)
                    predictions_all.extend(predictions.tolist())
                    labels_all.extend(labels.tolist())
                    
                avg_loss_val = np.mean(losses_val)
                avg_acc = accuracy_score(labels_all,predictions_all)
                print(f"Loss val: {avg_loss_val:0.4f}. Accuracy: {avg_acc:0.4f}")
                writer.add_scalar(tag="Val/Accuracy", scalar_value=avg_acc,global_step=epoch)
                
                cm = confusion_matrix(labels_all, predictions_all)
                plot_confusion_matrix(writer, cm, list(train_dataset.list_unique_labels), epoch)

                
                
                    
                checkpoint = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "fold": fold,
                    "confusion_matrix": cm,
                    "classes": classes,
                    "model_name": model_name,
                    "accuracy": avg_acc,
                }

                torch.save(checkpoint, os.path.join(model_fold_checkpoint_dir, "last.pt"))


                if avg_acc > best_acc:
                    best_acc = avg_acc
                    torch.save(checkpoint, os.path.join(model_fold_checkpoint_dir, "best.pt"))
                    print(f"Best model found at epoch {epoch}. Accuracy: {best_acc:.4f}") 
                       
                
                
            del model  # remove reference to model
            gc.collect()  # force garbage collection
            torch.cuda.empty_cache()  # release cache to CUDA
            torch.cuda.ipc_collect()  # release interprocess memory (if using multiple processes)
            
            
            
        
                    
                    
                
                
            
            
        
        # print(len(train_dataset), len(val_dataset))
        
        
    
    
    
    
    
    
    
if __name__ =='__main__':
    train()
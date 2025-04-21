import os
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import Resize, ToTensor, Compose
from torch.utils.data import Dataset, DataLoader
from PIL import Image



def get_list_img_label(path,classes):
    img_paths = []
    labels = []
    
    for class_id in os.listdir(path):
        
        if int(class_id) in classes:
            class_path = os.path.join(path,class_id)
            for img in os.listdir(class_path):
                img_path = os.path.join(class_path,img)
                img_paths.append(img_path)
                labels.append(class_id)
    
    return np.array(img_paths), np.array(labels)


class BVDLTW(Dataset):
    img_paths = []
    labels = []
    annotations = {14: "Vảy nến đỏ da toàn thân",
               12: "Vảy nến thông thường",
               13: "Vảy nến thể mủ",
               40: "Viêm da cơ địa bán cấp",
               43: "Viêm da cơ địa mạn tính",
               44: "Ung thư da tế bào đáy",
               45: "Ung thư da tế bào vảy",
               46: "Bệnh da khác",
               60: "Da không bị bệnh",
               4: "Ung thư hắc tố",
               3: "Viêm da cơ địa cấp tính",
               107: "Viêm da tiếp xúc dị ứng",
               109: "Viêm da ứ trệ",
               101: "Nấm sâu",
               102: "Lao da",
               103: "Dày sừng da dầu",
               104: "Vảy phấn hồng",
               105: "Vảy phấn đỏ nang lông",
               106: "Viêm da dầu",
               }
    
    def __init__(self,path=None,train=True,transform=None):
        
        
        
        if train:
            data_path = os.path.join(path, "train/raw")    
        else:
            data_path = os.path.join(path, "test/raw")
        for class_id in os.listdir(data_path):
            class_path = os.path.join(data_path,class_id)
            for img in os.listdir(class_path):
                img_path = os.path.join(class_path,img)
                self.__class__.img_paths.append(img_path)
                self.__class__.labels.append(class_id)
    
class SubsetBVDLTW(BVDLTW):
    def __init__(self, classes, path, train=True, transform=None):
        super().__init__(path, train, transform)
        self.transform = transform
        self.classes = classes
        self.labels = []
        self.img_paths = []
        for (img_path, label) in zip(super().img_paths,super().labels):
            if int(label) in self.classes:
                self.labels.append(label)
                self.img_paths.append(img_path)
                
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
               
        img = Image.open(self.img_paths[index]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[index]
        return img, label
      
      
      
class MyDataset(Dataset):
    def __init__(self, image_paths,labels, transform=None):
        self.image_paths = image_paths
        # self.labels = labels
        self.labels = []
        
        self.mapping = {}
        self.transform = transform
        set_labels = set(labels)
        list_unique_labels = list(set_labels)
        self.list_unique_labels = list_unique_labels
        # print(set_labels)
        for i in range(len(list_unique_labels)):
            self.mapping[i] = list_unique_labels[i]
            
        for label in labels:
            self.labels.append(list_unique_labels.index(label))
            
             
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[index]
        
        


      
if __name__ == "__main__":
    path = "./../../Datasets/BVDLTW/data_raw/all/raw"
    transform = Compose([
        ToTensor(),
        Resize(size=(250,250))
    ])
    classes = [44,45,4]
    # cancer_dataset = SubsetBVDLTW(classes=classes ,path=path,train=False,transform=transform)
    # print(f"number of image: {len(cancer_dataset)}")
    # print("----------------")
    # for (img,label) in cancer_dataset:
    #     print(f"image shape: {img.shape}, label: {label}")
    
    img_paths, labels_lists = get_list_img_label(path,classes)
    test_dataset = MyDataset(img_paths, labels_lists, transform)
    print(test_dataset.mapping)
    print(test_dataset.list_unique_labels)
    
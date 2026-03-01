import torch 
import pandas as pd 
import numpy as np
from torch.utils.data import Dataset , DataLoader 
from torchvision import transforms
from load_data import load_
from config import train_path , test_path
class train_config(Dataset):
    def __init__(self,X,y):
        super().__init__()

        self.features = np.array(X)
        if self.features.shape[1] == 784:
            self.features = self.features.reshape(-1,28,28)
        self.labels = np.array(y)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])


    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        image = self.features[idx] 
        label = self.labels[idx]
        image = self.transform(image)
        return image , torch.tensor(label,dtype=torch.long)
    
class test_config(Dataset):
    def __init__(self,X,y):
        super().__init__()

        self.features = np.array(X,dtype=np.uint8)
        if self.features.shape[1] == 784:
            self.features = self.features.reshape(-1,28,28)
        self.labels = np.array(y)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])


    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        image = self.features[idx] 
        label = self.labels[idx]
        image = self.transform(image)
        return image , torch.tensor(label,dtype=torch.long)

X_train , X_test , y_train , y_test = load_(train_path,test_path)
if __name__ == "__main__":
    print(type(X_train))
    print(X_train.dtype if hasattr(X_train, "dtype") else "No dtype")
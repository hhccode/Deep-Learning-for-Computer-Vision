import os
import pandas as pd
import numpy as np
from skimage import io
from skimage.color import gray2rgb
import torch
from torch.utils.data import Dataset


def rgb2gray(img):
    gray_img = 0.2125 * img[:, :, 0] + 0.7154 * img[:, :, 1] + 0.0721 * img[:, :, 2]

    return gray_img.astype(np.uint8)

class CelebA(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.image_path = os.path.join(self.root, "train")
        self.attr = pd.read_csv(os.path.join(self.root, "train.csv")).values
        self.file_name = sorted(os.listdir(self.image_path))
        self.len = len(self.file_name)
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image = io.imread(os.path.join(self.image_path, self.file_name[index]))
        
        if self.transform:
            image = self.transform(image)

        # Choose smiling as the attribute
        row = int(self.file_name[index].split(".")[0])
        label = self.attr[row][9]

        return image, label

class MNISTM(Dataset):
    def __init__(self, image_path, label_path, test_mode=False, transform=None, gray=False):
        self.image_path = image_path
        self.file_name = sorted(os.listdir(self.image_path))
        if not test_mode:
            self.label = pd.read_csv(label_path).values
        self.len = len(self.file_name)
        self.test_mode = test_mode
        self.transform = transform
        self.gray = gray
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        image = io.imread(os.path.join(self.image_path, self.file_name[index]))
        if self.gray:
            image = rgb2gray(image)

        if self.transform:
            image = self.transform(image)
        
        if not self.test_mode:
            return image, self.label[index, 1]
        else:
            return image, self.file_name[index]

class SVHN(Dataset):
    def __init__(self, image_path, label_path, test_mode=False, transform=None, gray=False):
        self.image_path = image_path
        self.file_name = sorted(os.listdir(self.image_path))
        if not test_mode:
            self.label = pd.read_csv(label_path).values
        self.len = len(self.file_name)
        self.test_mode = test_mode
        self.transform = transform
        self.gray = gray
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        image = io.imread(os.path.join(self.image_path, self.file_name[index]))
        if self.gray:
            image = rgb2gray(image)
        
        if self.transform:
            image = self.transform(image)
        
        if not self.test_mode:
            return image, self.label[index, 1]
        else:
            return image, self.file_name[index]

class USPS(Dataset):
    def __init__(self, image_path, label_path, test_mode=False, transform=None, gray=False):
        self.image_path = image_path
        self.file_name = sorted(os.listdir(self.image_path))
        if not test_mode:
            self.label = pd.read_csv(label_path).values
        self.len = len(self.file_name)
        self.test_mode = test_mode
        self.transform = transform
        self.gray = gray
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        image = io.imread(os.path.join(self.image_path, self.file_name[index]))
        if not self.gray:
            image = gray2rgb(image)
        
        if self.transform:
            image = self.transform(image)
        
        if not self.test_mode:
            return image, self.label[index, 1]
        else:
            return image, self.file_name[index]
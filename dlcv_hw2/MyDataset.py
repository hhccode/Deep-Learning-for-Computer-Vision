import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, image_path, label_path=None, transform=None, horizontal_flip=False, vertical_flip=False, grid_num=7):
        self.classes = {
            "plane": 0, "ship": 1, "storage-tank": 2, "baseball-diamond": 3, "tennis-court": 4,
            "basketball-court": 5, "ground-track-field": 6, "harbor": 7, "bridge": 8, "small-vehicle": 9,
            "large-vehicle": 10, "helicopter": 11, "roundabout": 12, "soccer-ball-field": 13, "swimming-pool": 14,
            "container-crane": 15
        }
        self.image_path = image_path
        self.label_path = label_path
        self.len = len(os.listdir(self.image_path))
        self.file_name = os.listdir(self.image_path)
        self.transform = transform
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.grid_num = grid_num

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        fname = self.file_name[index].split(".")[0]
        image_name = os.path.join(self.image_path, self.file_name[index])
        image = plt.imread(image_name)
        
        if self.label_path:
            label_name = os.path.join(self.label_path, fname + ".txt")
            label = self.label_processing(label_name)  

        if self.transform:
            image = self.transform(image)
        if self.horizontal_flip:
            image = torch.flip(image, [2])
        if self.vertical_flip:
            image = torch.flip(image, [1])

        if self.label_path:
            return image, label.astype(np.float32)
        else:
            return image

    def label_processing(self, file_name):
        grid_size = 512 / self.grid_num
        label = np.zeros((self.grid_num, self.grid_num, 26))

        with open(file_name, "r") as f:
            for line in f:
                line = line.strip().split()
                
                xmin = float(line[0])
                ymin = float(line[1])
                xmax = float(line[2])
                ymax = float(line[5])

                if self.horizontal_flip:
                    xmin = 512.0 - float(line[2])
                    xmax = 512.0 - float(line[0])
                if self.vertical_flip:
                    ymin = 512.0 - float(line[5])
                    ymax = 512.0 - float(line[1])


                class_ = self.classes[line[8]]

                w = (xmax - xmin) / 512
                h = (ymax - ymin) / 512

                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                x_grid = int(x_center / 512 * self.grid_num)
                y_grid = int(y_center / 512 * self.grid_num)
                x = x_center / 512 * self.grid_num - x_grid
                y = y_center / 512 * self.grid_num - y_grid
                
                paras = [x, y, w, h, 1.0, x, y, w, h, 1.0]
                
                for i, para in enumerate(paras):
                    label[y_grid][x_grid][i] = para
                label[y_grid][x_grid][10+class_] = 1
        
        return label
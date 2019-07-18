import os
from skimage import io
import torch
from torch.utils.data import Dataset
import reader

class TrimmedVideos(Dataset):
    def __init__(self, video_path, gt_path, transform, is_train):
        self.video_path = video_path
        self.gt = reader.getVideoList(data_path=gt_path)
        self.transform = transform
        self.is_train = is_train
        self.len = len(self.gt["Video_index"])
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        frames = reader.readShortVideo(
            video_path=self.video_path,
            video_category=self.gt["Video_category"][index],
            video_name=self.gt["Video_name"][index]
        )

        images = []
        for i in range(len(frames)):
            images.append(self.transform(frames[i]).unsqueeze(0))
        images = torch.cat(images)

        if self.is_train:
            label = int(self.gt["Action_labels"][index])

            return images, label
        return images

class FeaturesDataset(Dataset):
    def __init__(self, feature_path, label_path):
        self.features = torch.load(feature_path, map_location=torch.device("cpu"))
        self.labels = torch.load(label_path, map_location=torch.device("cpu"))
        self.len = len(self.features)

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index]

class FullLengthVideos(Dataset):
    def __init__(self, video_path, transform, is_train, gt_path=None):
        self.video_path = video_path
        self.categories = sorted(os.listdir(video_path))
        self.transform = transform
        self.is_train = is_train
        if self.is_train:
            self.gt_path = gt_path
        self.len = len(self.categories)

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        path = os.path.join(self.video_path, self.categories[index])
        frame_name = sorted(os.listdir(path))
        
        frames = []
        for i in range(len(frame_name)):
            frames.append(self.transform(io.imread(os.path.join(path, frame_name[i]))).unsqueeze(0))
        frames = torch.cat(frames)

        if self.is_train:
            label_path = os.path.join(self.gt_path, self.categories[index] + ".txt")

            labels = []
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    labels.append(int(line))
            labels = torch.tensor(labels)
        
            return frames, labels
        return frames
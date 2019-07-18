import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models import resnet50
import visdom
from datasets import TrimmedVideos, FeaturesDataset
from models import CNNVideoClassifier

def preprocess_dataset(video_path, gt_path, device, is_train):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = TrimmedVideos(
        video_path=video_path,
        gt_path=gt_path,
        transform=transform,
        is_train=True
    )
    loader = DataLoader(dataset, batch_size=1)

    model = resnet50(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    model.to(device)

    with torch.no_grad():
        for i, data in enumerate(loader):
            print("\r[{}/{}]".format(i+1, len(loader)), end="")
            inputs, targets = data
            inputs, targets = inputs.to(device).squeeze(0), targets.to(device)

            if i == 0:
                features = torch.mean(model(inputs).contiguous().view(inputs.size(0), -1), dim=0, keepdim=True)
                labels = targets
            else:
                features = torch.cat((features, torch.mean(model(inputs).contiguous().view(inputs.size(0), -1), dim=0, keepdim=True)))
                labels = torch.cat((labels, targets))
            
    if is_train:
        os.makedirs("./preprocess/p1/train", exist_ok=True)
        torch.save(features.cpu(), "./preprocess/p1/train/features.pkl")
        torch.save(labels.cpu(), "./preprocess/p1/train/labels.pkl")
    else:
        os.makedirs("./preprocess/p1/val", exist_ok=True)
        torch.save(features.cpu(), "./preprocess/p1/val/features.pkl")
        torch.save(labels.cpu(), "./preprocess/p1/val/labels.pkl")

def main():
    vis = visdom.Visdom()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size = 64
    EPOCH = 100

    # preprocess_dataset(
    #     video_path="./hw4_data/TrimmedVideos/video/train",
    #     gt_path="./hw4_data/TrimmedVideos/label/gt_train.csv",
    #     device=device,
    #     is_train=True
    # )
    # preprocess_dataset(
    #     video_path="./hw4_data/TrimmedVideos/video/valid",
    #     gt_path="./hw4_data/TrimmedVideos/label/gt_valid.csv",
    #     device=device,
    #     is_train=False
    # )
    
    train_dataset = FeaturesDataset(feature_path="./preprocess/p1/train/features.pkl", label_path="./preprocess/p1/train/labels.pkl")
    val_dataset = FeaturesDataset(feature_path="./preprocess/p1/val/features.pkl", label_path="./preprocess/p1/val/labels.pkl")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    train_steps = len(train_loader)
    val_steps = len(val_loader)

    model = CNNVideoClassifier(feature_dim=2048)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    for epoch in range(EPOCH):
        
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for i, data in enumerate(train_loader):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            print("\rEpoch: [{}/{}] | Step: [{}/{}] | Loss={:.5f}".format(epoch+1, EPOCH, i+1, train_steps, loss.item()), end="")
            
            predict = torch.max(outputs, 1)[1]
            train_loss += loss.item()
            train_acc += np.sum((predict == targets).cpu().numpy())

        model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                predict = torch.max(outputs, 1)[1]
                val_loss += loss.item()
                val_acc += np.sum((predict == targets).cpu().numpy())
        
        vis.line(
            X=np.array([epoch+1]),
            Y=np.array([train_loss/train_steps, val_loss/val_steps]).reshape(1, 2),
            win="P2 Loss", update="append", 
            opts={  
                "legend": ["train loss", "val loss"],
                "xlabel": "Epoch",
                "ylabel": "loss",
                "title": "Training loss curve"
            }
        )

        vis.line(
            X=np.array([epoch+1]),
            Y=np.array([train_acc/train_size, val_acc/val_size]).reshape(1, 2),
            win="P1 Acc", update="append", 
            opts={  
                "legend": ["train acc", "val acc"],
                "xlabel": "Epoch",
                "ylabel": "accuracy",
                "title": "Training accuracy curve"
            }
        )

        torch.save(model.state_dict(), "./checkpoints/p1/{}.ckpt".format(epoch+1))

if __name__ == "__main__":
    os.makedirs("./checkpoints/p1", exist_ok=True)
    main()
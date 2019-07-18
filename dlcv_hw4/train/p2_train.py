import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models import resnet50
import visdom
from datasets import TrimmedVideos
from models import RNNVideoClassifier

def collate_fn(batch):
    inputs = []
    targets = []
    lengths = []

    for i in range(len(batch)):
        inputs.append(batch[i][0])
        targets.append(batch[i][1])
        lengths.append(batch[i][0].size(0))

    inputs = torch.cat(inputs)
    targets = torch.tensor(targets)
    lengths = torch.tensor(lengths)

    return inputs, targets, lengths

def preprocess_dataset(video_path, gt_path, device, is_train):
    trans_without_aug = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = TrimmedVideos(
        video_path=video_path,
        gt_path=gt_path,
        transform=trans_without_aug,
        is_train=True
    )

    if is_train:
        trans_with_aug = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(1.0),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = dataset.__add__(TrimmedViedos(
            video_path=video_path,
            gt_path=gt_path,
            transform=trans_with_aug,
            is_train=True
        ))

    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    model = resnet50(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    model.to(device)

    features = []
    labels = []
    seq_length = []

    with torch.no_grad():
        for i, data in enumerate(loader):
            print("\r[{}/{}]".format(i+1, len(loader)), end="")
            inputs, targets, lengths = data
            inputs = inputs.to(device)
            
            # Shape: N x 2048 
            video_feat = model(inputs).contiguous().view(inputs.size(0), -1).cpu()

            pos = 0
            for j in range(lengths.size(0)):
                features.append(video_feat[pos:pos+lengths[j]])
                pos += lengths[j]
                labels.append(targets[j])
                seq_length.append(lengths[j])

    if is_train:
        os.makedirs("./preprocess/p2/train", exist_ok=True)

        with open("./preprocess/p2/train/features.pkl", "wb") as f:
            pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open("./preprocess/p2/train/labels.pkl", "wb") as f:
            pickle.dump(labels, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open("./preprocess/p2/train/seq_length.pkl", "wb") as f:
            pickle.dump(seq_length, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        os.makedirs("./preprocess/p2/val", exist_ok=True)

        with open("./preprocess/p2/val/features.pkl", "wb") as f:
            pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open("./preprocess/p2/val/labels.pkl", "wb") as f:
            pickle.dump(labels, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open("./preprocess/p2/val/seq_length.pkl", "wb") as f:
            pickle.dump(seq_length, f, protocol=pickle.HIGHEST_PROTOCOL)
    
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
    
    with open("./preprocess/p2/train/features.pkl", "rb") as f:
        train_features = pickle.load(f)
    with open("./preprocess/p2/train/labels.pkl", "rb") as f:
        train_labels = torch.tensor(pickle.load(f))
    with open("./preprocess/p2/train/seq_length.pkl", "rb") as f:
        train_seq_length = torch.tensor(pickle.load(f))

    with open("./preprocess/p2/val/features.pkl", "rb") as f:
        val_features = pickle.load(f)
    with open("./preprocess/p2/val/labels.pkl", "rb") as f:
        val_labels = torch.tensor(pickle.load(f))
    with open("./preprocess/p2/val/seq_length.pkl", "rb") as f:
        val_seq_length = torch.tensor(pickle.load(f))
    
    train_size = len(train_features)
    val_size = len(val_features)
    train_steps = int(np.ceil(train_size / batch_size))
    val_steps = int(np.ceil(val_size / batch_size))
    
    model = RNNVideoClassifier(input_dim=2048, hidden_dim=1024)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    for epoch in range(EPOCH):
        
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        index = np.random.permutation(train_size)
        shuf_train_features = [train_features[idx] for idx in index] 
        shuf_train_labels = train_labels[index]
        shuf_train_lengths = train_seq_length[index]

        for i in range(0, train_size, batch_size):
            inputs = [data.to(device) for data in shuf_train_features[i:i+batch_size]]
            targets = shuf_train_labels[i:i+batch_size].to(device)
            lens = shuf_train_lengths[i:i+batch_size]
            
            optimizer.zero_grad()
            
            outputs = model(inputs, lens)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            print("\rEpoch: [{}/{}] | Step: [{}/{}] | Loss={:.5f}".format(epoch+1, EPOCH, (i//batch_size)+1, train_steps, loss.item()), end="")
            
            predict = torch.max(outputs, 1)[1]
            train_loss += loss.item()
            train_acc += np.sum((predict == targets).cpu().numpy())
                    
        model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for i in range(0, val_size, batch_size):
                inputs = [data.to(device) for data in val_features[i:i+batch_size]]
                targets = val_labels[i:i+batch_size].to(device)
                lens = val_seq_length[i:i+batch_size]
                        
                outputs = model(inputs, lens)
                loss = criterion(outputs, targets)

                predict = torch.max(outputs, 1)[1]
                val_loss += loss.item()
                val_acc += np.sum((predict == targets).cpu().numpy())
                
        vis.line(
            X=np.array([epoch+1]),
            Y=np.array([train_loss/train_steps, val_loss/val_steps]).reshape(1, 2),
            win="Loss", update="append", 
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
            win="Acc", update="append", 
            opts={  
                "legend": ["train acc", "val acc"],
                "xlabel": "Epoch",
                "ylabel": "accuracy",
                "title": "Training accuracy curve"
            }
        )

        torch.save(model.state_dict(), "./checkpoints/p2/{}.ckpt".format(epoch+1))

if __name__ == "__main__":
    os.makedirs("./checkpoints/p2", exist_ok=True)
    main()
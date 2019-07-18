import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models import resnet50
import visdom
from datasets import FullLengthVideos
from models import Seq2Seq

def collate_fn(batch):
    inputs = []
    targets = []
    lengths = []

    for i in range(len(batch)):
        inputs.append(batch[i][0])
        targets.append(batch[i][1])
        lengths.append(batch[i][0].size(0))

    inputs = torch.cat(inputs)
    targets = torch.cat(targets)
    lengths = torch.tensor(lengths)

    return inputs, targets, lengths

def preprocess_dataset(video_path, gt_path, device, is_train):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = FullLengthVideos(
        video_path=video_path,
        transform=transform,
        is_train=True,
        gt_path=gt_path
    )
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    
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
            
            video_feat = []
            sample_num = inputs.size(0)
            
            for j in range(0, sample_num, 128):
                frames = inputs[j:j+128].to(device)
                # Shape: 128 x 2048 
                video_feat.append(model(frames).contiguous().view(frames.size(0), -1).cpu())
                
            video_feat = torch.cat(video_feat)

            features.append(video_feat)
            labels.append(targets)
            seq_length.append(lengths[0])

    if is_train:
        os.makedirs("./preprocess/p3/train", exist_ok=True)

        with open("./preprocess/p3/train/features.pkl", "wb") as f:
            pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open("./preprocess/p3/train/labels.pkl", "wb") as f:
            pickle.dump(labels, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open("./preprocess/p3/train/seq_length.pkl", "wb") as f:
            pickle.dump(seq_length, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        os.makedirs("./preprocess/p3/val", exist_ok=True)

        with open("./preprocess/p3/val/features.pkl", "wb") as f:
            pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open("./preprocess/p3/val/labels.pkl", "wb") as f:
            pickle.dump(labels, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open("./preprocess/p3/val/seq_length.pkl", "wb") as f:
            pickle.dump(seq_length, f, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    vis = visdom.Visdom()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size = 3
    EPOCH = 100

    # preprocess_dataset(
    #     video_path="./hw4_data/FullLengthVideos/videos/train",
    #     gt_path="./hw4_data/FullLengthVideos/labels/train",
    #     device=device,
    #     is_train=True
    # )
    # preprocess_dataset(
    #     video_path="./hw4_data/FullLengthVideos/videos/valid",
    #     gt_path="./hw4_data/FullLengthVideos/labels/valid",
    #     device=device,
    #     is_train=False
    # )

    with open("./preprocess/p3/train/features.pkl", "rb") as f:
        train_features = pickle.load(f)
    with open("./preprocess/p3/train/labels.pkl", "rb") as f:
        train_labels = pickle.load(f)
    with open("./preprocess/p3/train/seq_length.pkl", "rb") as f:
        train_seq_length = torch.tensor(pickle.load(f))

    with open("./preprocess/p3/val/features.pkl", "rb") as f:
        val_features = pickle.load(f)
    with open("./preprocess/p3/val/labels.pkl", "rb") as f:
        val_labels = pickle.load(f)
    with open("./preprocess/p3/val/seq_length.pkl", "rb") as f:
        val_seq_length = torch.tensor(pickle.load(f))
    

    train_size = len(train_seq_length)
    val_size = len(val_seq_length)
    train_steps = int(np.ceil(train_size / batch_size))
    video_size = 512

    model = Seq2Seq(input_dim=2048, hidden_dim=1024)
    model.load_state_dict(torch.load("./model/p2.ckpt"))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    for epoch in range(EPOCH):
        
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        index = np.random.permutation(train_size)
        shuf_train_features = [train_features[idx] for idx in index] 
        shuf_train_labels = [train_labels[idx] for idx in index] 
        shuf_train_lengths = train_seq_length[index]

        for i in range(0, train_size, batch_size):
            
            inputs, targets = [], []
            for bs in range(batch_size):
                if i + bs >= train_size:
                    break
                frame_num = shuf_train_features[i+bs].size(0)
                frame_idx = sorted(np.random.choice(frame_num, size=video_size, replace=False))
                
                inputs.append(shuf_train_features[i+bs][frame_idx].to(device))
                targets.append(shuf_train_labels[i+bs][frame_idx])
            
            # Shape: batch size x video size (3, 512)
            targets = torch.stack(targets).to(device)
            lens = shuf_train_lengths[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(inputs, torch.tensor([video_size for _ in range(len(inputs))]))
            
            loss = 0.0
            for j in range(len(inputs)):
                loss += criterion(outputs[j], targets[j])
            loss /= len(inputs)
            loss.backward()
            optimizer.step()

            print("\rEpoch: [{}/{}] | Step: [{}/{}] | Loss={:.5f}".format(epoch+1, EPOCH, (i//batch_size)+1, train_steps, loss.item()), end="")
            
            predict = torch.max(outputs, 2)[1]
            train_loss += loss.item()
            train_acc += np.sum((predict == targets).cpu().numpy())
            
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        total = 0

        with torch.no_grad():
            for i in range(val_size):
                inputs = [val_features[i].to(device)]
                targets = val_labels[i].to(device)
                lens = val_seq_length[i:i+1]
                        
                outputs = model(inputs, lens).squeeze(0)
                loss = criterion(outputs, targets)

                predict = torch.max(outputs, 1)[1]
                val_loss += loss.item()
                val_acc += np.sum((predict == targets).cpu().numpy())
                total += targets.size(0)

        train_loss /= train_steps
        train_acc /= video_size * train_size
        val_loss /= val_size
        val_acc /= total

        vis.line(
            X=np.array([epoch+1]),
            Y=np.array([train_loss, val_loss]).reshape(1, 2),
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
            Y=np.array([train_acc, val_acc]).reshape(1, 2),
            win="Acc", update="append", 
            opts={  
                "legend": ["train acc", "val acc"],
                "xlabel": "Epoch",
                "ylabel": "accuracy",
                "title": "Training accuracy curve"
            }
        )

        torch.save(model.state_dict(), "./checkpoints/p3/{}.ckpt".format(epoch+1))

if __name__ == "__main__":
    os.makedirs("./checkpoints/p3", exist_ok=True)
    main()
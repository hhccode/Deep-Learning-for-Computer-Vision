import os
import sys
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models import resnet50
from datasets import FullLengthVideos
from models import Seq2Seq

def collate_fn(batch):
    inputs = []
    lengths = []

    for i in range(len(batch)):
        inputs.append(batch[i])
        lengths.append(batch[i].size(0))

    inputs = torch.cat(inputs)
    lengths = torch.tensor(lengths)

    return inputs, lengths

def preprocess_dataset(video_path, device):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = FullLengthVideos(
        video_path=video_path,
        transform=transform,
        is_train=False
    )
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    
    model = resnet50(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    model.to(device)

    features = []
    seq_length = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            print("\r[{}/{}]".format(i+1, len(loader)), end="")
            inputs, lengths = data
            
            video_feat = []
            sample_num = inputs.size(0)
            
            for j in range(0, sample_num, 128):
                frames = inputs[j:j+128].to(device)
                # Shape: 128 x 2048 
                video_feat.append(model(frames).contiguous().view(frames.size(0), -1).cpu())
                
            video_feat = torch.cat(video_feat)

            features.append(video_feat)
            seq_length.append(lengths[0])
    
    return features, torch.tensor(seq_length), dataset.categories

def main(argv):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    print("\nStart preprocessing the data")
    features, seq_length, categories = preprocess_dataset(video_path=argv[1], device=device)
    print("\nFinish preprocessing the data")

    size = len(features)
    
    model = Seq2Seq(input_dim=2048, hidden_dim=1024)
    model.load_state_dict(torch.load("./model/p3.ckpt"))
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i in range(size):
            with open(os.path.join(argv[2], categories[i] + ".txt"), "w", newline="") as csvfile:
                writer = csv.writer(csvfile)

                inputs = [features[i].to(device)]
                lens = seq_length[i:i+1]
                        
                outputs = model(inputs, lens).squeeze(0)

                predict = torch.max(outputs, 1)[1]
                
                for j in range(predict.size(0)):
                    writer.writerow([predict[j].item()])

if __name__ == "__main__":
    main(sys.argv)
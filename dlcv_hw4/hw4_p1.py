import os
import sys
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models import resnet50
from datasets import TrimmedVideos
from models import CNNVideoClassifier

def collate_fn(batch):
    inputs = []
    lengths = []

    for i in range(len(batch)):
        inputs.append(batch[i])
        lengths.append(batch[i].size(0))

    inputs = torch.cat(inputs)
    lengths = torch.tensor(lengths)
    
    return inputs, lengths

def main(argv):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    batch_size = 4

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = TrimmedVideos(
        video_path=argv[1],
        gt_path=argv[2],
        transform=transform,
        is_train=False
    )
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    steps = len(loader)

    backbone = resnet50(True)
    backbone = nn.Sequential(*list(backbone.children())[:-1])
    backbone.to(device)
    backbone.eval()

    model = CNNVideoClassifier(feature_dim=2048)
    model.load_state_dict(torch.load("./model/p1.ckpt")) # Epoch 37
    model.to(device)
    model.eval()

    with torch.no_grad():
        with open(os.path.join(argv[3], "p1_valid.txt"), "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            for i, data in enumerate(loader):
                print("\rStep: [{}/{}]".format(i+1, steps), end="")
                
                inputs, lengths = data
                inputs = inputs.to(device)
                
                outputs = backbone(inputs).contiguous().view(inputs.size(0), -1)
                
                pos = 0
                features = []
                for j in range(lengths.size(0)):
                    features.append(torch.mean(outputs[pos:pos+lengths[j], :], dim=0, keepdim=True))
                    pos += lengths[j]
                
                features = torch.cat(features)
                
                outputs = model(features)
                predict = torch.max(outputs, 1)[1]

                for j in range(predict.size(0)):
                    writer.writerow([predict[j].item()])
            
            print("")
    
if __name__ == "__main__":
    main(sys.argv)
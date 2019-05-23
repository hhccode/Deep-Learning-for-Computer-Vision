import os
import sys
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from datasets import MNISTM, USPS
from models import GTA_F, GTA_C

def main(argv):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(32),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    if argv[2] == "mnistm":
        dataset = MNISTM(image_path=argv[1], label_path=None, test_mode=True, transform=transform)
        source = "usps"
        target = "mnistm"
    elif argv[2] == "usps":
        dataset = USPS(image_path=argv[1], label_path=None, test_mode=True, transform=transform)
        source = "svhn"
        target = "usps"
    
    batch_size = 64
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    F = GTA_F(filter_num=64)
    C = GTA_C(channel_num=128)
    F.load_state_dict(torch.load("./model/gta/F/{}_{}.pkl".format(source, target)))
    C.load_state_dict(torch.load("./model/gta/C/{}_{}.pkl".format(source, target)))
    F.eval()
    C.eval()
    F.to(device)
    C.to(device)

    with torch.no_grad():
        with open(argv[3], "w", newline="") as csvfile:

            writer = csv.writer(csvfile)
            writer.writerow(["image_name", "label"])

            for _, data in enumerate(loader):
                inputs, file_name = data
                inputs = inputs.to(device)
            
                outputs = C(F(inputs))

                predict = torch.max(outputs, 1)[1].cpu().numpy()

                for i in range(len(file_name)):
                    writer.writerow([file_name[i], predict[i]])

if __name__ == "__main__":
    main(sys.argv)
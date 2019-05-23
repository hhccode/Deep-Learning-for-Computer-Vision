import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from datasets import MNISTM, SVHN, USPS
from models import CNN

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    mnistm_dataset = MNISTM(image_path="./hw3_data/digits/mnistm/test", label_path=None, test_mode=True, transform=T.ToTensor())
    svhn_dataset = SVHN(image_path="./hw3_data/digits/svhn/test", label_path=None, test_mode=True, transform=T.ToTensor())
    usps_dataset = USPS(image_path="./hw3_data/digits/usps/test", label_path=None, test_mode=True, transform=T.ToTensor())
    
    mnistm_loader = DataLoader(mnistm_dataset, batch_size=64, shuffle=False)
    svhn_loader = DataLoader(svhn_dataset, batch_size=64, shuffle=False)
    usps_loader = DataLoader(usps_dataset, batch_size=64, shuffle=False)
    targets_loaders = [mnistm_loader, svhn_loader, usps_loader]

    model = CNN(filter_num=64)
    model.to(device)

    sources = ["usps", "mnistm", "svhn"]
    targets = ["mnistm", "svhn", "usps"]

    with torch.no_grad():
        for i in range(len(targets_loaders)):
            
            model.load_state_dict(torch.load("trained_on_{}.pkl".format(targets[i])))
            model.eval()


            with open("{}_{}.csv".format(targets[i], targets[i]), "w", newline="") as csvfile:

                writer = csv.writer(csvfile)
                writer.writerow(["image_name", "label"])

                for j, data in enumerate(targets_loaders[i]):
                    inputs, file_name = data
                    inputs = inputs.to(device)
                
                    outputs = model(inputs)

                    predict = torch.max(outputs, 1)[1].cpu().numpy()

                    for k in range(len(file_name)):
                        writer.writerow([file_name[k], predict[k]])
        
        for i in range(len(targets_loaders)):
            
            model.load_state_dict(torch.load("trained_on_{}.pkl".format(sources[i])))
            model.eval()


            with open("{}_{}.csv".format(sources[i], targets[i]), "w", newline="") as csvfile:

                writer = csv.writer(csvfile)
                writer.writerow(["image_name", "label"])

                for j, data in enumerate(targets_loaders[i]):
                    inputs, file_name = data
                    inputs = inputs.to(device)
                
                    outputs = model(inputs)

                    predict = torch.max(outputs, 1)[1].cpu().numpy()

                    for k in range(len(file_name)):
                        writer.writerow([file_name[k], predict[k]])
                
    
if __name__ == "__main__":
    main()
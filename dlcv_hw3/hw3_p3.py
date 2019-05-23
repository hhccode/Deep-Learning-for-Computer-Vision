import os
import sys
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from datasets import MNISTM, SVHN, USPS
from models import DANN

def main(argv):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    

    if argv[2] == "mnistm":
        dataset = MNISTM(image_path=argv[1], label_path=None, test_mode=True, transform=T.ToTensor())
        source = "usps"
        target = "mnistm"
    elif argv[2] == "svhn":
        dataset = SVHN(image_path=argv[1], label_path=None, test_mode=True, transform=T.ToTensor())
        source = "mnistm"
        target = "svhn"
    elif argv[2] == "usps":
        dataset = USPS(image_path=argv[1], label_path=None, test_mode=True, transform=T.ToTensor())
        source = "svhn"
        target = "usps"
    
    batch_size = 64
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = DANN(filter_num=64)
    model.to(device)

    with torch.no_grad():

        model.load_state_dict(torch.load("./model/{}_{}.pkl".format(source, target)))
        model.eval()

        with open(argv[3], "w", newline="") as csvfile:

            writer = csv.writer(csvfile)
            writer.writerow(["image_name", "label"])

            for _, data in enumerate(loader):
                inputs, file_name = data
                inputs = inputs.to(device)
            
                outputs, _ = model(inputs, 0)

                predict = torch.max(outputs, 1)[1].cpu().numpy()

                for i in range(len(file_name)):
                    writer.writerow([file_name[i], predict[i]])
                

if __name__ == "__main__":
    main(sys.argv)
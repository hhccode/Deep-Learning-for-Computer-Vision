import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as T
from datasets import MNISTM, SVHN, USPS
from models import DANN

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    mnistm_dataset = MNISTM(image_path="./hw3_data/digits/mnistm/train", label_path="./hw3_data/digits/mnistm/train.csv", transform=T.ToTensor())
    svhn_dataset = SVHN(image_path="./hw3_data/digits/svhn/train", label_path="./hw3_data/digits/svhn/train.csv", transform=T.ToTensor())
    usps_dataset = USPS(image_path="./hw3_data/digits/usps/train", label_path="./hw3_data/digits/usps/train.csv", transform=T.ToTensor())

    batch_size = 64
    mnistm_loader = DataLoader(mnistm_dataset, batch_size=batch_size, shuffle=True)
    svhn_loader = DataLoader(svhn_dataset, batch_size=batch_size, shuffle=True)
    usps_loader = DataLoader(usps_dataset, batch_size=batch_size, shuffle=True)
    
    source_loaders = [usps_loader, mnistm_loader, svhn_loader]
    target_loaders = [mnistm_loader, svhn_loader, usps_loader]

    model = DANN(filter_num=64)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=1e-4)

    EPOCH = 20

    paths = ["usps-mnistm", "mnistm-svhn", "svhn-usps"]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


    for i in range(len(source_loaders)):
        
        steps = min(len(source_loaders[i]), len(target_loaders[i]))
        model.reset_weights()
        model.train()
        
        for epoch in range(EPOCH):
            
            train_loss = 0.0

            source_data = iter(source_loaders[i])
            target_data = iter(target_loaders[i])
            
            for step in range(steps):
                optimizer.zero_grad()

                p = float(step + epoch * steps) / (EPOCH * steps)
                lamb = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
                

                # Source data
                inputs, classes = next(source_data)
                inputs, classes = inputs.to(device), classes.to(device)
                bs = inputs.size(0)
                domains = torch.zeros(bs).long().to(device)

                class_outputs, domain_outputs = model(inputs, lamb)

                source_class_loss = criterion(class_outputs, classes)
                source_domain_loss = criterion(domain_outputs, domains)
            

                # Target data
                inputs, _ = next(target_data)
                inputs = inputs.to(device)
                bs = inputs.size(0)
                domains = torch.ones(bs).long().to(device)

                _, domain_outputs = model(inputs, lamb)

                target_domain_loss = criterion(domain_outputs, domains)


                loss = source_class_loss + source_domain_loss + target_domain_loss
                loss.backward()
                optimizer.step()
    
                train_loss += loss.item()

                print("\rTask: [{}/3], Epoch: [{}/{}], Step: [{}/{}], Loss: {:.5f}".format(i, epoch, EPOCH, step, steps, loss.item()), end="")


            print("\n\nSource-Target: {}".format(paths[i]))
            print("Finish epoch {}\n".format(epoch))
            torch.save(model.state_dict(), os.path.join(paths[i], "epoch{}-{:.5f}.pkl".format(str(epoch).zfill(3), train_loss/steps)))
        
        

if __name__ == "__main__":
    main()
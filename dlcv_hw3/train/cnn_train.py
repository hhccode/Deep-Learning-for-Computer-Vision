import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as T
from datasets import MNISTM, SVHN, USPS
from models import CNN

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    mnistm_dataset = MNISTM(image_path="./hw3_data/digits/mnistm/train", label_path="./hw3_data/digits/mnistm/train.csv", transform=T.ToTensor())
    svhn_dataset = SVHN(image_path="./hw3_data/digits/svhn/train", label_path="./hw3_data/digits/svhn/train.csv", transform=T.ToTensor())
    usps_dataset = USPS(image_path="./hw3_data/digits/usps/train", label_path="./hw3_data/digits/usps/train.csv", transform=T.ToTensor())
    
    batch_size = 64
    mnistm_loader = DataLoader(mnistm_dataset, batch_size=batch_size, shuffle=True)
    svhn_loader = DataLoader(svhn_dataset, batch_size=batch_size, shuffle=True)
    usps_loader = DataLoader(usps_dataset, batch_size=batch_size, shuffle=True)
    loaders = [mnistm_loader, svhn_loader, usps_loader]

    model = CNN(filter_num=64)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=1e-4)

    EPOCH = 20

    paths = ["mnistm", "svhn", "usps"]
    
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

    for i in range(len(loaders)):

        steps = len(loaders[i])
        model.reset_weights()
        model.train()

        for epoch in range(EPOCH):
            
            train_loss = 0.0
            train_acc = 0.0
            
            for j, data in enumerate(loaders[i]):
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                predict = torch.max(outputs, 1)[1]
                acc = np.mean((targets == predict).cpu().numpy())
                
                train_loss += loss.item()
                train_acc += acc

                print("\r{} Epoch {}, loss = {:.5f}, acc = {:.5f}".format(paths[i], epoch, loss.item(), acc), end="")
            
        
            print("\n\n\nFinish epoch {}".format(epoch))
            print("{} Loss = {:.5f}, Acc = {:.5f}\n\n".format(paths[i], train_loss/steps, train_acc/steps))

            torch.save(model.state_dict(), os.path.join(paths[i], "epoch{}-loss{:.5f}-acc{:.5f}.pkl".format(epoch, train_loss/steps, train_acc/steps)))

        print("Finish {}".format(paths[i]))

if __name__ == "__main__":
    main()
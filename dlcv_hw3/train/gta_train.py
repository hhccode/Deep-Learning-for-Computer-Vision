import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as T
from datasets import MNISTM, SVHN, USPS
from models import GTA

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(32),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    mnistm_dataset = MNISTM(image_path="./hw3_data/digits/mnistm/train", label_path="./hw3_data/digits/mnistm/train.csv", transform=transform)
    svhn_dataset = SVHN(image_path="./hw3_data/digits/svhn/train", label_path="./hw3_data/digits/svhn/train.csv", transform=transform)
    usps_dataset = USPS(image_path="./hw3_data/digits/usps/train", label_path="./hw3_data/digits/usps/train.csv", transform=transform)

    batch_size = 64
    mnistm_loader = DataLoader(mnistm_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    svhn_loader = DataLoader(svhn_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    usps_loader = DataLoader(usps_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    source_loaders = [usps_loader, svhn_loader]
    target_loaders = [mnistm_loader, usps_loader]

    model = GTA(latent_dim=512, batch_size=batch_size, device=device)
    model.move_to_device()

    EPOCH = 100

    paths = ["usps-mnistm", "svhn-usps"]   

    for i in range(len(source_loaders)):
        if not os.path.exists(paths[i]):
            os.makedirs(paths[i])
            os.makedirs(os.path.join(paths[i], "C"))
            os.makedirs(os.path.join(paths[i], "F"))

        steps = min(len(source_loaders[i]), len(target_loaders[i]))
        model.reset_all_weights()
        
        for epoch in range(EPOCH):
            
            D_loss = 0.0
            G_loss = 0.0
            C_loss = 0.0
            F_loss = 0.0

            source_data = iter(source_loaders[i])
            target_data = iter(target_loaders[i])
            
            for step in range(steps):
                # Source data
                source_inputs, source_labels = next(source_data)
                
                source_labels_onehot = torch.zeros((source_inputs.size(0), 10), dtype=torch.float)
                for j in range(source_inputs.size(0)):
                    source_labels_onehot[j][source_labels[j]] = 1

                source_inputs, source_labels, source_labels_onehot = source_inputs.to(device), source_labels.to(device), source_labels_onehot.to(device)
                
                # Target data
                target_inputs, _ = next(target_data)
                target_inputs = target_inputs.to(device)
                

                d_loss = model.train_D(source_inputs, source_labels, source_labels_onehot, target_inputs)
                g_loss = model.train_G(source_labels)
                c_loss = model.train_C(source_inputs, source_labels)
                f_loss = model.train_F(source_labels)

                print("\rTask: [{}/3], Epoch: [{}/{}], Step: [{}/{}], D_loss: {:.5f}, G_loss: {:.5f}, C_loss: {:.5f}, F_loss: {:.5f}".format(i+1, epoch+1, EPOCH, step+1, steps, d_loss, g_loss, c_loss, f_loss), end="")

                D_loss += d_loss
                G_loss += g_loss
                C_loss += c_loss
                F_loss += f_loss
            
            print("")

            print("\n\nEpoch {}: D_loss = {:.5f}, G_loss = {:.5f}, C_loss = {:.5f}, F_loss = {:.5f}".format(epoch+1, D_loss/steps, G_loss/steps, C_loss/steps, F_loss/steps))
            
            model.save_C(os.path.join(paths[i], "C", "{}.pkl".format(str(epoch+1).zfill(2))))
            model.save_F(os.path.join(paths[i], "F", "{}.pkl".format(str(epoch+1).zfill(2))))

if __name__ == "__main__":
    main()
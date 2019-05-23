import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as T
from datasets import MNISTM, SVHN
from models import ADDA

def main():
    pretrain = False
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    mnistm_dataset = MNISTM(image_path="./hw3_data/digits/mnistm/train", label_path="./hw3_data/digits/mnistm/train.csv", transform=T.ToTensor())
    svhn_dataset = SVHN(image_path="./hw3_data/digits/svhn/train", label_path="./hw3_data/digits/svhn/train.csv", transform=T.ToTensor())

    batch_size = 64
    mnistm_loader = DataLoader(mnistm_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    svhn_loader = DataLoader(svhn_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = ADDA(batch_size=batch_size, device=device)

    EPOCH = 20

    if not os.path.exists("mnistm-svhn"):
        os.makedirs("mnistm-svhn")
        os.makedirs(os.path.join("mnistm-svhn", "classifier"))
        os.makedirs(os.path.join("mnistm-svhn", "src_encoder"))
        os.makedirs(os.path.join("mnistm-svhn", "tgt_encoder"))

        
    #model.pretrain_src(mnistm_loader)
    model.load_pretrain(encoder_path="./model/adda/src_encoder.pkl", classifier_path="./model/adda/classifier.pkl")
    model.tgt_encoder.train()
    model.discriminator.train()
    model.src_encoder.train()
    model.tgt_encoder.to(device)
    model.discriminator.to(device)
    model.src_encoder.to(device)

    steps = min(len(mnistm_loader), len(svhn_loader))
    
    for epoch in range(EPOCH):
        
        source_data = iter(mnistm_loader)
        target_data = iter(svhn_loader)
        
        D_loss = 0.0
        Domain_acc = 0.0
        Target_loss = 0.0

        for step in range(steps):
            # Source data
            source_inputs, _ = next(source_data)
            source_inputs = source_inputs.to(device)
    
            # Target data
            target_inputs, _ = next(target_data)
            target_inputs = target_inputs.to(device)
            

            d_loss, acc = model.train_discri(source_inputs, target_inputs)
            tgt_loss = model.train_tgt(target_inputs)

            print("\rEpoch:[{}/{}], Step:[{}/{}], Domain acc:{:5f}, D loss:{:.5f}, Target loss:{:.5f}".format(epoch+1, EPOCH, step+1, steps, acc, d_loss, tgt_loss), end="")
            
            D_loss += d_loss
            Domain_acc += acc
            Target_loss += tgt_loss
        
        print("\r")
        print("Epoch {}, Domain acc:{:5f}, D loss:{:.5f}, Target loss:{:.5f}".format(epoch+1, Domain_acc/steps, D_loss/steps, Target_loss/steps))
        model.save_tgt_encoder("./mnistm-svhn/tgt_encoder/{}.pkl".format(epoch+1))

if __name__ == "__main__":
    main()
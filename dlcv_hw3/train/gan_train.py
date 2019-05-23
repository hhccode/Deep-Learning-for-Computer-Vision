import os
import time
from skimage import io
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from models import GAN
from datasets import CelebA


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dataset = CelebA(root="./hw3_data/face", transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    size = len(dataloader)
    
    model = GAN(latent_dim=100, batch_size=128, device=device)
    model.move_to_device()
    
    
    EPOCH = 100
    D_losses = []
    G_losses = []

    if not os.path.exists("./gan"):
        os.makedirs("./gan")
        os.makedirs("./gan/ckpt")
        os.makedirs("./gan/ckpt/D")
        os.makedirs("./gan/ckpt/G")
        os.makedirs("./gan/imgs")


    for epoch in range(EPOCH):
        D_loss = 0.0
        G_loss = 0.0
        start = time.time()


        for i, data in enumerate(dataloader):
            inputs, _ = data
            inputs = inputs.to(device)

            
            d_loss = model.train_D(inputs)
            g_loss = model.train_G()

            print("\rDiscriminator loss = {:.5f}, Generator loss = {:.5f}".format(d_loss, g_loss), end="")
    
            D_loss += d_loss
            G_loss += g_loss
        
        print("\n\n\nEpoch {}: D_loss = {:.5f}, G_loss = {:.5f}\n\n".format(epoch, D_loss/size, G_loss/size))
        
        D_losses.append(D_loss/size)
        G_losses.append(G_loss/size)

        #if (epoch + 1) % 5 == 0:
        model.save_D(fname="epoch{}-DL{:.5f}-GL{:.5f}".format(epoch, D_losses[-1], G_losses[-1]))
        print("Finish saving Discriminator")

        model.save_G(fname="epoch{}-DL{:.5f}-GL{:.5f}".format(epoch, D_losses[-1], G_losses[-1]))
        print("Finish saving Generator")

        model.save_image(fname=epoch)
        print("Finish saving generated images")
        
        print("Epoch {} is finished, taking {} seconds.".format(epoch, time.time()-start))
    
    plt.figure(1)
    plt.plot([i+1 for i in range(EPOCH)], D_losses, label="Discriminator")
    plt.plot([i+1 for i in range(EPOCH)], G_losses, label="Generator") 
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    plt.savefig("./gan/loss.png")


if __name__ == "__main__":
    main()
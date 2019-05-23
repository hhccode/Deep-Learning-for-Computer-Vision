import os
import sys
import torch
import torch.nn as nn
from models import GAN

def main(argv):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if not os.path.exists(argv[1]):
        os.makedirs(argv[1])
    
    model = GAN(latent_dim=100, batch_size=128, device=device)
    model.move_to_device()
    model.G.load_state_dict(torch.load("./model/gan-G.pkl"))

    fname = os.path.join(argv[1], "fig1_2.jpg")
    
    model.save_image(fname)
    
if __name__ == "__main__":
    main(sys.argv)
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from sklearn.manifold import TSNE
from datasets import MNISTM, SVHN
from models import ADDA_Encoder


def main(argv):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    mnistm_dataset = MNISTM(image_path="./hw3_data/digits/mnistm/test", label_path="./hw3_data/digits/mnistm/test.csv", test_mode=False, transform=T.ToTensor())
    svhn_dataset = SVHN(image_path="./hw3_data/digits/svhn/test", label_path="./hw3_data/digits/svhn/test.csv", test_mode=False, transform=T.ToTensor())
    
    source_dataset = mnistm_dataset
    target_dataset = svhn_dataset
    source = "mnistm"
    target = "svhn"

    batch_size = 64
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=False)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False)

    

    tsne = TSNE(n_components=2, init="pca", random_state=0)

    X = np.array([]).reshape(0, 3200)
    Y_class = np.array([], dtype=np.int16).reshape(0,)
    Y_domain = np.array([], dtype=np.int16).reshape(0,)

    with torch.no_grad():
        steps = len(source_loader)
        src_encoder = ADDA_Encoder()
        src_encoder.load_state_dict(torch.load("./model/adda/src_encoder.pkl"))
        src_encoder.to(device)
        src_encoder.eval()
        for i, data in enumerate(source_loader):
            inputs, classes = data
            inputs= inputs.to(device)

            outputs = src_encoder(inputs).contiguous().view(inputs.size(0), -1).cpu().numpy()
            classes = classes.numpy()

            X = np.vstack((X, outputs))
            Y_class = np.concatenate((Y_class, classes))
            Y_domain = np.concatenate((Y_domain, np.array([0 for _ in range(inputs.size(0))], dtype=np.int16)))
            
            print("Source stpes: [{}/{}]".format(i, steps))
        
        print(X.shape)
        print(Y_class.shape)
        print(Y_domain.shape)

        steps = len(target_loader)
        tgt_encoder = ADDA_Encoder()
        tgt_encoder.load_state_dict(torch.load("./model/adda/tgt_encoder.pkl"))
        tgt_encoder.to(device)
        tgt_encoder.eval()
        for i, data in enumerate(target_loader):
            inputs, classes = data
            inputs= inputs.to(device)

            outputs = tgt_encoder(inputs).contiguous().view(inputs.size(0), -1).cpu().numpy()
            classes = classes.numpy()

            X = np.vstack((X, outputs))
            Y_class = np.concatenate((Y_class, classes))
            Y_domain = np.concatenate((Y_domain, np.array([1 for _ in range(inputs.size(0))], dtype=np.int16)))
            
            print("Target stpes: [{}/{}]".format(i, steps))
        
        print(X.shape)
        print(Y_class.shape)
        print(Y_domain.shape)

    X_tsne = tsne.fit_transform(X)
    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)


    color = ['r', 'g', 'b', 'k', 'gold', 'm', 'c', 'orange', 'cyan', 'pink']
    class_color = [color[label] for label in Y_class]
    domain_color = [color[label] for label in Y_domain]


    plt.figure(1, figsize=(8, 8))
    plt.scatter(X_norm[:, 0], X_norm[:, 1], c=class_color, s=1)
    plt.savefig("./{}_{}_class.png".format(source, target))
    plt.close("all")


    plt.figure(2, figsize=(8, 8))
    plt.scatter(X_norm[:, 0], X_norm[:, 1], c=domain_color, s=1)
    plt.savefig("./{}_{}_domain.png".format(source, target))
    plt.close("all")

if __name__ == "__main__":
    main(sys.argv)
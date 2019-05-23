import sys
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from datasets import SVHN
from models import ADDA_Encoder, ADDA_Classifier

def main(argv):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    dataset = SVHN(image_path=argv[1], label_path=None, test_mode=True, transform=T.ToTensor())
    source = "mnistm"
    target = "svhn"

    batch_size = 64
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    encoder = ADDA_Encoder()
    encoder.load_state_dict(torch.load("./model/adda/tgt_encoder.pkl"))
    encoder.to(device)
    encoder.eval()
    classifier = ADDA_Classifier()
    classifier.load_state_dict(torch.load("./model/adda/classifier.pkl"))
    classifier.to(device)
    classifier.eval()


    with torch.no_grad():
        with open(argv[3], "w", newline="") as csvfile:

            writer = csv.writer(csvfile)
            writer.writerow(["image_name", "label"])

            for _, data in enumerate(loader):
                inputs, file_name = data
                inputs = inputs.to(device)
            
                outputs = classifier(encoder(inputs))

                predict = torch.max(outputs, 1)[1].cpu().numpy()

                for i in range(len(file_name)):
                    writer.writerow([file_name[i], predict[i]])
        
if __name__ == "__main__":
    main(sys.argv)
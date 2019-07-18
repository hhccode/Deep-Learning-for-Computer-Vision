import pickle
import numpy as np
import torch
from torch.nn.utils import rnn
import visdom
from MulticoreTSNE import MulticoreTSNE as TSNE
from models import RNNVideoClassifier

def CNN(vis, seed):
    features = torch.load("./preprocess/p1/val/features.pkl").numpy()
    labels = torch.load("./preprocess/p1/val/labels.pkl").numpy()
    
    
    # TSNE
    tsne = TSNE(n_components=2, n_iter=3000, verbose=1, random_state=seed, n_jobs=20)
    X = tsne.fit_transform(features)
    X_min, X_max = X.min(0), X.max(0)
    X_tsne = (X - X_min) / (X_max - X_min)


    vis.scatter(X=X_tsne, Y=labels+1, win="CNN", 
                opts={  
                    "legend": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                    "markersize": 5,
                    "title": "CNN-based features"
                })

def RNN(vis, seed):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size = 64

    with open("./preprocess/p2/val/features.pkl", "rb") as f:
        features = pickle.load(f)
    with open("./preprocess/p2/val/labels.pkl", "rb") as f:
        labels = torch.tensor(pickle.load(f))
    with open("./preprocess/p2/val/seq_length.pkl", "rb") as f:
        seq_length = torch.tensor(pickle.load(f))
    
    val_size = len(features)

    model = RNNVideoClassifier(input_dim=2048, hidden_dim=1024)
    model.load_state_dict(torch.load("./model/p2.ckpt"))
    model.to(device)
    model.eval()

    embeddings = np.array([]).reshape(0, 1024)
    with torch.no_grad():
        for i in range(0, val_size, batch_size):
            inputs = [data.to(device) for data in features[i:i+batch_size]]
            lens = seq_length[i:i+batch_size]
                        
            # Sort the lengths in descending order
            idx = torch.argsort(-lens)
            # Keep the original index
            desort_idx = torch.argsort(idx)
            # Sort x and length in descending order        
            x = [inputs[i] for i in idx]
            lens = lens[idx]

            x_padded = rnn.pad_sequence(x, batch_first=True)
            x_packed = rnn.pack_padded_sequence(x_padded, lens, batch_first=True)
            gru_outputs, _ = model.gru(x_packed)
            gru_outputs, _ = rnn.pad_packed_sequence(gru_outputs, batch_first=True)
        
            outputs = gru_outputs[torch.arange(gru_outputs.size(0)), lens-1].cpu().numpy()
            outputs = outputs[desort_idx].reshape(-1, 1024)

            embeddings = np.concatenate((embeddings, outputs))

    
    # TSNE
    tsne = TSNE(n_components=2, n_iter=3000, verbose=1, random_state=seed, n_jobs=20)
    X = tsne.fit_transform(embeddings)
    X_min, X_max = X.min(0), X.max(0)
    X_tsne = (X - X_min) / (X_max - X_min)


    vis.scatter(X=X_tsne, Y=labels+1, win="RNN", 
                opts={  
                    "legend": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                    "markersize": 5,
                    "title": "RNN-based features"
                })
            
                
if __name__ == "__main__":
    vis = visdom.Visdom()
    seed = 0
    CNN(vis, seed)
    RNN(vis, seed)
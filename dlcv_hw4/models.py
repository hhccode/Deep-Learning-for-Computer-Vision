import torch
import torch.nn as nn
from torch.nn.utils import rnn

class CNNVideoClassifier(nn.Module):
    def __init__(self, feature_dim):
        super(CNNVideoClassifier, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 11)
        )

    def forward(self, x):
        return self.fc(x)

class RNNVideoClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RNNVideoClassifier, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden_dim, 11)
        )
        

    def forward(self, x, length):
        # Sort the lengths in descending order
        idx = torch.argsort(-length)
        # Keep the original index
        desort_idx = torch.argsort(idx)
        # Sort x and length in descending order        
        x = [x[i] for i in idx]
        length = length[idx]

        x_padded = rnn.pad_sequence(x, batch_first=True)
        x_packed = rnn.pack_padded_sequence(x_padded, length, batch_first=True)
        gru_outputs, _ = self.gru(x_packed)
        gru_outputs, _ = rnn.pad_packed_sequence(gru_outputs, batch_first=True)
        
        gru_outputs = gru_outputs[torch.arange(gru_outputs.size(0)), length-1]
        
        outputs =  self.fc(gru_outputs)
        outputs = outputs[desort_idx]
        
        return outputs

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Seq2Seq, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden_dim, 11)
        )
        

    def forward(self, x, length):
        # Sort the lengths in descending order
        idx = torch.argsort(-length)
        # Keep the original index
        desort_idx = torch.argsort(idx)
        # Sort x and length in descending order        
        x = [x[i] for i in idx]
        length = length[idx]

        x_padded = rnn.pad_sequence(x, batch_first=True)
        x_packed = rnn.pack_padded_sequence(x_padded, length, batch_first=True)
        gru_outputs, _ = self.gru(x_packed)
        gru_outputs, _ = rnn.pad_packed_sequence(gru_outputs, batch_first=True)
        
        outputs =  self.fc(gru_outputs)
        outputs = outputs[desort_idx]
  
        return outputs
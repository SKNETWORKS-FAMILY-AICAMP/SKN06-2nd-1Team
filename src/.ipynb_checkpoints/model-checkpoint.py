import torch.nn as nn
import torch

class DropoutModel(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.lr1 = nn.Sequential(nn.Linear(12, 128), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.lr2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.lr3 = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.lr4 = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.out_block = nn.Linear(16, 1)

    def forward(self, X):
        X = nn.Flatten()(X)
        X = self.lr1(X)
        X = self.lr2(X)
        X = self.lr3(X)
        X = self.lr4(X)
        output = torch.sigmoid(self.out_block(X))
        return output
import torch
import torch.nn as nn 

class AmazonNeuralNetwork(nn.Module):
    def __init__(self, in_features=26, n_classes=15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)


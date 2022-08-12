import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['mlp', 'over_mlp']


class MLP(nn.Module):
    def __init__(self, input_dim=32, num_classes=10, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def mlp(num_classes, **kwargs):
    return MLP(num_classes, **kwargs)

class OverMLP(nn.Module):

    def __init__(self, input_dim=32, num_classes=10, over_factor=5, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32*over_factor)
        self.fc2 = nn.Linear(32*over_factor, 32*over_factor)
        self.fc3 = nn.Linear(64*over_factor, 64*over_factor)
        self.fc4 = nn.Linear(64*over_factor, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def over_mlp(num_classes, **kwargs):
    return OverMLP(num_classes, **kwargs)

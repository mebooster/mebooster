import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['incep_net', 'over_incep_net']


class IncepNet(nn.Module):
    """
    Classifying Garments from Fashion-MNIST Dataset Through CNNs
    """
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1, 1) #input_channel, output_channel, kernal, stride, padding
        self.conv2 = nn.Conv2d(6, 6, 3, 1, 1)
        self.conv3 = nn.Conv2d(6, 12, 3, 1, 1)
        self.conv4 = nn.Conv2d(12, 12, 3, 1, 1)
        self.conv5 = nn.Conv2d(12, 32, 3, 1, 1)
        self.fc1 = nn.Linear(7*7*32, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)#28
        x = F.dropout(x, 0.2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))#14
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv5(x))#7
        x = F.dropout(x, 0.2)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def incep_net(num_classes, **kwargs):
    return IncepNet(num_classes, **kwargs)

class OverIncepNet(nn.Module):
    """
    lenet5-[17]
    """

    def __init__(self, num_classes=10, over_factor=5, **kwargs):
        super().__init__()
        self.over_factor = over_factor
        self.conv1 = nn.Conv2d(1, 6*self.over_factor, 3, 1, 1)  # input_channel, output_channel, kernal, stride, padding
        self.conv2 = nn.Conv2d(6*self.over_factor, 6*self.over_factor, 3, 1, 1)
        self.conv3 = nn.Conv2d(6*self.over_factor, 12*self.over_factor, 3, 1, 1)
        self.conv4 = nn.Conv2d(12*self.over_factor, 12*self.over_factor, 3, 1)
        self.conv5 = nn.Conv2d(12*self.over_factor, 32*self.over_factor, 3, 1)
        self.fc1 = nn.Linear(7 * 7 * 32*self.over_factor, 120*self.over_factor)
        self.fc2 = nn.Linear(120*self.over_factor, 84*self.over_factor)
        self.fc3 = nn.Linear(84*self.over_factor, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)  # 28
        x = F.dropout(x, 0.2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))  # 14
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv5(x))  # 7
        x = F.dropout(x, 0.2)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def over_incep_net(num_classes, **kwargs):
    return OverIncepNet(num_classes, **kwargs)
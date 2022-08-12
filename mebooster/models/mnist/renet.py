import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['renet', 'renet1', 'rsnet', 'e_rsnet']


class ReNet(nn.Module):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1,
                               padding=1)  # input_channel, output_channel, kernal, stride, padding
        self.conv2 = nn.Conv2d(32, 16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(12 * 12 * 16, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 24
        x = F.relu(self.conv2(x))  # 8x8
        x = F.max_pool2d(x, 2, 2)  # 4x4
        x = x.view(-1, 12 * 12 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def renet(num_classes, **kwargs):
    return ReNet(num_classes, **kwargs)

class ReNet1(nn.Module):
    """A simple MNIST network

    Source: https://github.com/pytorch/examples/blob/master/mnist/main.py
    """
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=4, stride=4, padding=0) #9
        # self.conv2 = nn.Conv2d(16, 16, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(7*7*16, 16)
        # self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x)) #28
        # x = F.relu(self.conv2(x)) #28
        x = F.relu(self.conv3(x))
        #x = F.max_pool2d(x, 3, 3)#
        x = x.view(-1, 7*7*16)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def renet1(num_classes, **kwargs):
    return ReNet1(num_classes, **kwargs)


class RSNet(nn.Module):
    # A simpler MNIST network
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=4, stride=4) #input_channel, output_channel, stride, padding
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=4, stride=3)
        # self.conv3 = nn.Conv2d(10, 10, 5, 1)
        self.fc1 = nn.Linear(10, 10) #4*5
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x)) #25x25
        x = F.relu(self.conv2(x)) #2x2
        x = F.max_pool2d(x, 2, 2) #1x1
        x = x.view(-1, 10) #4*5
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def rsnet(num_classes, **kwargs):
    return RSNet(num_classes, **kwargs)


class ERSNet(nn.Module):
    # A simpler MNIST network
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=4, stride=4) #input_channel, output_channel, stride, padding
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=4, stride=3)
        # self.conv3 = nn.Conv2d(10, 10, 5, 1)
        self.fc1 = nn.Linear(10, 10) #4*5
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x)) #25x25
        x = F.relu(self.conv2(x)) #2x2
        x = F.max_pool2d(x, 2, 2) #1x1
        x = x.view(-1, 10) #4*5
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def e_rsnet(num_classes, **kwargs):
    return ERSNet(num_classes, **kwargs)
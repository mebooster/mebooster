import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['lenet_tl', 'over_lenet_tl', 'over_lenet5', 'over_another_lenet']

class Another_Lenet(nn.Module):
    """A simple MNIST network

        Source: https://github.com/pytorch/examples/blob/main/mnist_hogwild/main.py
        """
    def __init__(self, num_classes, over_factor=5):
        super(Another_Lenet, self).__init__()
        self.over_factor=over_factor
        self.conv1 = nn.Conv2d(1, 6*self.over_factor, kernel_size=5, stride=1)#10
        self.conv2 = nn.Conv2d(6*self.over_factor, 20*self.over_factor, kernel_size=5, stride=1)#10
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320*self.over_factor, 150*self.over_factor)
        self.fc2 = nn.Linear(150 * self.over_factor, 50 * self.over_factor)
        self.fc3 = nn.Linear(50*self.over_factor, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320*self.over_factor)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)

        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def over_another_lenet(num_classes, over_factor, **kwargs):
    return Another_Lenet(num_classes, over_factor, **kwargs)

class LeNet(nn.Module):
    """
    lenet5-[17]
    """
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1) #input_channel, output_channel, kernal, stride, padding
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.fc1 = nn.Linear(4*4*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def midx(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 16)
        return x

def lenet_tl(num_classes, **kwargs):
    return LeNet(num_classes, **kwargs)

class OverLeNet(nn.Module):
    """
    lenet5-[17]
    """
    def __init__(self, num_classes=10, over_factor=5, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1) #input_channel, output_channel, kernal, stride, padding
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.fc1 = nn.Linear(4*4*16, 120*over_factor)
        self.fc2 = nn.Linear(120*over_factor, 84*over_factor)
        self.fc3 = nn.Linear(84*over_factor, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def over_lenet_tl(num_classes, **kwargs):
    return OverLeNet(num_classes, **kwargs)

class OverLeNet5(nn.Module):
    """
    lenet5-[17]
    """
    def __init__(self, num_classes=10, over_factor=5, **kwargs):
        super().__init__()
        self.over_factor=over_factor
        self.conv1 = nn.Conv2d(1, int(6*over_factor), 5, 1) #input_channel, output_channel, kernal, stride, padding
        self.conv2 = nn.Conv2d(int(6*over_factor), int(16*over_factor), 5, 1)
        self.fc1 = nn.Linear(int(4*4*16*over_factor), int(120*over_factor))
        self.fc2 = nn.Linear(int(120*over_factor), int(84*over_factor))
        self.fc3 = nn.Linear(int(84*over_factor), num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, int(4*4*16* self.over_factor))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def over_lenet5(num_classes, over_factor=5, **kwargs):
    return OverLeNet5(num_classes,over_factor=over_factor, **kwargs)
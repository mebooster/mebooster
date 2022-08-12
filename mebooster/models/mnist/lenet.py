import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['lenet','nn2', 'snet', 'over_snet', 'over_snet_low', 'over_snet_high',
           'over_nn2', 'nn4', 'over_nn4', 'gaussian_nn', 'over_gaussian_nn',
           'gaussian_cnn', 'over_gaussian_cnn', 'snet_midx', 'over_snet_4k3s', 'snet_4k3s']


class LeNet(nn.Module):
    """A simple MNIST network

    Source: https://github.com/pytorch/examples/blob/master/mnist/main.py
    """
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5, 1) #input_channel, output_channel, kernal, stride, padding
        self.conv2 = nn.Conv2d(10, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def lenet(num_classes, **kwargs):
    return LeNet(num_classes, **kwargs)

#snet for mnist

class OverSNet(nn.Module):
    """A simpler MNIST network

    """
    def __init__(self, num_classes=10, over_factor=5, **kwargs):
        super().__init__()
        self.over_factor = over_factor
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=int(10 * over_factor),
                               kernel_size=4, stride=4) #input_channel, output_channel, stride, padding
        self.conv2 = nn.Conv2d(in_channels=int(10 * over_factor), out_channels=int(15 * over_factor),
                               kernel_size=5, stride=2)
        self.fc1 = nn.Linear(int(15 * over_factor), int(10 * over_factor)) #4*5
        self.fc2 = nn.Linear(int(10 * over_factor), int(10 * over_factor))
        self.fc3 = nn.Linear(int(10 * over_factor), num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # print(x.shape)
        # x = F.max_pool2d(x, 4, 4)
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.max_pool2d(x, 2, 2)
        # print(x.shape)
        # x = F.relu(self.conv3(x))
        x = x.view(-1, (15*self.over_factor)) #4*5
        #print("x", x.shape)
        #print("fc1,", self.fc1.weight.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = F.relu(self.fc2(x))
        # print(x.shape)
        x = self.fc3(x)
        return x

    def midx(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.shape[0], -1) #10*5
        x = F.relu(self.fc1(x))
        return x

def over_snet(num_classes, **kwargs):
    return OverSNet(num_classes, **kwargs)

class OverSNetHigh(nn.Module):
    """A simpler MNIST network

    """
    def __init__(self, num_classes=10, over_factor=5, **kwargs):
        super().__init__()
        self.over_factor = over_factor
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=int(10 * over_factor),
                               kernel_size=4, stride=4) #input_channel, output_channel, stride, padding
        self.conv2 = nn.Conv2d(in_channels=int(10 * over_factor), out_channels=int(15 * over_factor),
                               kernel_size=5, stride=2)
        self.fc1 = nn.Linear(int(15 * over_factor), int(10 * over_factor)) #4*5
        self.fc2 = nn.Linear(int(10 * over_factor), int(10 * over_factor))
        self.fc3 = nn.Linear(int(10 * over_factor), num_classes)

    def forward(self, x):
        #x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.max_pool2d(x, 2, 2)
        # print(x.shape)
        # x = F.relu(self.conv3(x))
        x = x.view(-1, (15*self.over_factor)) #4*5
        #print("x", x.shape)
        #print("fc1,", self.fc1.weight.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = F.relu(self.fc2(x))
        # print(x.shape)
        x = self.fc3(x)
        return x

    def midx(self, x):
        x = F.relu(self.conv1(x))
        # print(x.shape)
        # x = F.max_pool2d(x, 4, 4)
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.max_pool2d(x, 2, 2)
        # print(x.shape)
        # x = F.relu(self.conv3(x))
        x = x.view(-1, (15*self.over_factor)) #10*5
        x = F.relu(self.fc1(x))
        return x

def over_snet_high(num_classes, **kwargs):
    return OverSNetHigh(num_classes, **kwargs)

class OverSNetLow(nn.Module):
    """A simpler MNIST network

    """
    def __init__(self, num_classes=10, over_factor=5, **kwargs):
        super().__init__()
        self.over_factor = over_factor
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=int(10 * over_factor),
                               kernel_size=4, stride=4) #input_channel, output_channel, stride, padding
        self.conv2 = nn.Conv2d(in_channels=int(10 * over_factor), out_channels=int(15 * over_factor),
                               kernel_size=5, stride=2)
        self.fc1 = nn.Linear(int(15 * over_factor), int(10 * over_factor)) #4*5
        self.fc2 = nn.Linear(int(10 * over_factor), int(10 * over_factor))
        self.fc3 = nn.Linear(int(10 * over_factor), num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        """
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.max_pool2d(x, 2, 2)
        # print(x.shape)
        # x = F.relu(self.conv3(x))
        x = x.view(-1, (15*self.over_factor)) #4*5
        #print("x", x.shape)
        #print("fc1,", self.fc1.weight.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = F.relu(self.fc2(x))
        # print(x.shape)
        x = self.fc3(x)
        """
        return x

    def midx(self, x):
        x = F.relu(self.conv1(x))
        # print(x.shape)
        # x = F.max_pool2d(x, 4, 4)
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.max_pool2d(x, 2, 2)
        # print(x.shape)
        # x = F.relu(self.conv3(x))
        x = x.view(-1, (15*self.over_factor)) #10*5
        x = F.relu(self.fc1(x))
        return x

def over_snet_low(num_classes, **kwargs):
    return OverSNetLow(num_classes, **kwargs)

class SNet(nn.Module):
    # A simpler MNIST network

    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=4, stride=4) #input_channel, output_channel, stride, padding
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=15, kernel_size=5, stride=2)
        # self.conv3 = nn.Conv2d(10, 10, 5, 1)
        self.fc1 = nn.Linear(15, 10) #4*5
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x)) #25x25
        x = F.relu(self.conv2(x)) #2x2
        x = F.max_pool2d(x, 2, 2) #1x1
        x = x.view(-1, 15) #4*5
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def midx(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.shape[0], -1)  # 10*5
        x = F.relu(self.fc1(x))
        return x

def snet(num_classes, **kwargs):
    return SNet(num_classes, **kwargs)

class SNetMidx(nn.Module):
    # A simpler MNIST network

    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=4, stride=4) #input_channel, output_channel, stride, padding
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=15, kernel_size=5, stride=2)
        # self.conv3 = nn.Conv2d(10, 10, 5, 1)
        self.fc1 = nn.Linear(15, 10) #4*5
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x)) #25x25
        x = F.relu(self.conv2(x)) #2x2
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2) #1x1
        x = x.view(-1, 15) #4*5
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def midx(self, x):
        x = F.relu(self.conv1(x))
        return x

    def last_half(self, x):
        x = F.relu(self.conv2(x))  # 2x2
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)  # 1x1
        x = x.view(-1, 15)  # 4*5
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def snet_midx(num_classes, **kwargs):
    return SNetMidx(num_classes, **kwargs)

#2-layer nns for mnist

class TwoLayerNet(nn.Module):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        # self.conv1 = nn.Conv2d(1, 20, 5, 1) #input_channel, output_channel, stride, padding
        # self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(1*28*28, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, 2, 2)
        x = x.view(-1,1*28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def nn2(num_classes, **kwargs):
    return TwoLayerNet(num_classes, **kwargs)

class OverTwoLayerNet(nn.Module):
    def __init__(self, num_classes=10, over_factor=1, **kwargs):
        super().__init__()
        # self.conv1 = nn.Conv2d(1, 20, 5, 1) #input_channel, output_channel, stride, padding
        # self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(1*28*28, int(100*over_factor))
        self.fc2 = nn.Linear(int(100*over_factor), num_classes)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, 2, 2)
        x = x.view(-1,1*28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def over_nn2(num_classes, over_factor, **kwargs):
    return OverTwoLayerNet(num_classes, over_factor, **kwargs)

#deep linear nns for mnist

class OverFourLayerNet(nn.Module):
    def __init__(self, num_classes=10, over_factor=1, **kwargs):
        super().__init__()
        # self.conv1 = nn.Conv2d(1, 20, 5, 1) #input_channel, output_channel, stride, padding
        # self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(1*28*28, int(20*over_factor))

        self.fc2 = nn.Linear(int(20*over_factor), int(20 * over_factor))
        self.fc3 = nn.Linear(int(20 * over_factor), int(20 * over_factor))
        self.fc4 = nn.Linear(int(20 * over_factor), num_classes)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, 2, 2)
        x = x.view(-1,1*28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def over_nn4(num_classes, over_factor, **kwargs):
    return OverFourLayerNet(num_classes, over_factor, **kwargs)

class FourLayerNet(nn.Module):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        # self.conv1 = nn.Conv2d(1, 20, 5, 1) #input_channel, output_channel, stride, padding
        # self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(1*28*28, 20)
        self.fc2 = nn.Linear(20, 20)

        self.fc3 = nn.Linear(20, 20)

        self.fc4 = nn.Linear(20, num_classes)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 1*28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def nn4(num_classes, **kwargs):
    return FourLayerNet(num_classes, **kwargs)

#for gaussian

class TwoLayerNetGaussian(nn.Module):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__()
        # self.conv1 = nn.Conv2d(1, 20, 5, 1) #input_channel, output_channel, stride, padding
        # self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(20, 10) #20, 10 6, 5
        self.fc2 = nn.Linear(10, 10) #10, 10 5, 5
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, num_classes) #10 5

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = x.view(-1, 30)

        x = F.relu(self.fc1(x))
        # x = torch.sigmoid(self.fc1(x))
        # x = F.relu(self.fc1(x)) * F.relu(self.fc1(x)) #squared Relu
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # x = torch.sum(x, dim=1)
        return x

def gaussian_nn(num_classes, **kwargs):
    return TwoLayerNetGaussian(num_classes, **kwargs)

class OverTwoLayerNetGaussian(nn.Module):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__()
        # self.conv1 = nn.Conv2d(1, 20, 5, 1) #input_channel, output_channel, stride, padding
        # self.conv2 = nn.Conv2d(20, 50, 5, 1)
        over_factor=5
        self.fc1 = nn.Linear(20, 10*over_factor)
        self.fc2 = nn.Linear(10*over_factor, 10*over_factor)
        self.fc3 = nn.Linear(10*over_factor, 10*over_factor)
        self.fc4 = nn.Linear(10*over_factor, num_classes)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = x.view(-1, 30)

        x = F.relu(self.fc1(x))
        # x = torch.sigmoid(self.fc1(x))
        # x = F.relu(self.fc1(x)) * F.relu(self.fc1(x)) #squared Relu
        # x = self.fc2(x)

        # x = torch.sum(x, dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def over_gaussian_nn(num_classes, **kwargs):
    return OverTwoLayerNetGaussian(num_classes, **kwargs)

#cnn for guassian (batch_size, c, h, w)
class SnetGaussian(nn.Module):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 3, 2) #input_channel, output_channel, kernel_size, stride, padding
        self.conv2 = nn.Conv2d(3, 5, 2, 2)
        self.fc1 = nn.Linear(5, 5) # [5, 1 ,1]
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, num_classes) #10 5

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, 2, 2)
        x = x.view(batch_size, -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def gaussian_cnn(num_classes, **kwargs):
    return SnetGaussian(num_classes, **kwargs)

#cnn for guassian (batch_size, c, h, w)
class OverSnetGaussian(nn.Module):
    def __init__(self, num_classes=2, over_factor=3, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 3, 2) #input_channel, output_channel, stride, padding
        self.conv2 = nn.Conv2d(3, 5, 3, 2)
        self.fc1 = nn.Linear(5, 5*over_factor) # [5, 1 ,1]
        self.fc2 = nn.Linear(5*over_factor, 5*over_factor)
        self.fc3 = nn.Linear(5*over_factor, num_classes) #10 5

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, 2, 2)
        x = x.view(batch_size, -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def over_gaussian_cnn(num_classes, **kwargs):
    return OverSnetGaussian(num_classes, **kwargs)

class OverSNet_4k3s(nn.Module):
    """A simpler MNIST network

    """
    def __init__(self, num_classes=10, over_factor=5, **kwargs):
        super().__init__()
        self.over_factor = over_factor
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=int(10 * over_factor),
                               kernel_size=4, stride=3) #input_channel, output_channel, stride, padding
        self.conv2 = nn.Conv2d(in_channels=int(10 * over_factor), out_channels=int(10 * over_factor),
                               kernel_size=5, stride=2)
        self.fc1 = nn.Linear(int(10 * over_factor), int(10 * over_factor)) #4*5
        self.fc2 = nn.Linear(int(10 * over_factor), int(10 * over_factor))
        self.fc3 = nn.Linear(int(10 * over_factor), num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # print(x.shape)
        # x = F.max_pool2d(x, 4, 4)
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.max_pool2d(x, 3, 3)
        # print(x.shape)
        # x = F.relu(self.conv3(x))
        x = x.view(-1, (10*self.over_factor)) #4*5
        #print("x", x.shape)
        #print("fc1,", self.fc1.weight.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = F.relu(self.fc2(x))
        # print(x.shape)
        x = self.fc3(x)
        return x

    def midx(self, x):
        x = self.conv1(x)
        # print(x.shape)
        # x = F.max_pool2d(x, 4, 4)
        # print(x.shape)
        # x = F.relu(self.conv2(x))
        # # print(x.shape)
        # x = F.max_pool2d(x, 3, 3)
        # # print(x.shape)
        # # x = F.relu(self.conv3(x))
        # x = x.view(-1, (10*self.over_factor)) #10*5
        return x

    def midx2(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        # # print(x.shape)
        # x = F.max_pool2d(x, 3, 3)
        # # print(x.shape)
        # # x = F.relu(self.conv3(x))
        # x = x.view(-1, (10*self.over_factor)) #10*5
        return x

def over_snet_4k3s(num_classes, **kwargs):
    return OverSNet_4k3s(num_classes, **kwargs)

class SNet_4k3s(nn.Module):
    # A simpler MNIST network

    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=4, stride=3) #input_channel, output_channel, stride, padding
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5, stride=2)
        # self.conv3 = nn.Conv2d(10, 10, 5, 1)
        self.fc1 = nn.Linear(10, 10) #4*5
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x)) #25x25
        # print(x.shape)
        # x = F.max_pool2d(x, 4, 4) #kernel_size, stride 6x6
        # print(x.shape)
        x = F.relu(self.conv2(x)) #2x2
        # print(x.shape)
        x = F.max_pool2d(x, 3, 3) #1x1
        # print(x.shape)
        # x = F.relu(self.conv3(x))
        x = x.view(-1, 10) #4*5
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = F.relu(self.fc2(x))
        # print(x.shape)
        x = self.fc3(x)
        return x

    def midx(self, x):
        x = self.conv1(x)
        # print(x.shape)
        # x = F.max_pool2d(x, 4, 4)
        # print(x.shape)
        # x = F.relu(self.conv2(x))
        # print(x.shape)
        # x = F.max_pool2d(x, 3, 3)
        # print(x.shape)
        # x = F.relu(self.conv3(x))
        # x = x.view(-1, 10) #10*5
        return x

    def midx2(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        # print(x.shape)
        # x = F.max_pool2d(x, 3, 3)
        # print(x.shape)
        # x = F.relu(self.conv3(x))
        # x = x.view(-1, 10) #10*5
        return x

def snet_4k3s(num_classes, **kwargs):
    return SNet_4k3s(num_classes, **kwargs)
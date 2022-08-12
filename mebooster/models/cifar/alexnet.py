'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['alexnet', 'cifar_cnn', 'over_cifar_cnn']

class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def alexnet(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(**kwargs)
    return model


class OverLayerNet(nn.Module):
    def __init__(self, num_classes=10, over_factor=1, **kwargs):
        super().__init__()
        # self.conv1 = nn.Conv2d(1, 20, 5, 1) #input_channel, output_channel, stride, padding
        # self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(1024, int(50*over_factor))

        self.fc2 = nn.Linear(int(50*over_factor), int(50 * over_factor))
        self.fc3 = nn.Linear(int(50 * over_factor), int(50 * over_factor))
        self.fc4 = nn.Linear(int(50 * over_factor), num_classes)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, 2, 2)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def over_cifar_cnn(num_classes, over_factor, **kwargs):
    return OverLayerNet(num_classes, over_factor, **kwargs)

class LayerNet(nn.Module):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        # self.conv1 = nn.Conv2d(1, 20, 5, 1) #input_channel, output_channel, stride, padding
        # self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(1024, 50)
        self.fc2 = nn.Linear(50, 50)

        self.fc3 = nn.Linear(50, 50)

        self.fc4 = nn.Linear(50, num_classes)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, 2, 2)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # print("x.shape", x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def cifar_cnn(num_classes, **kwargs):
    return LayerNet(num_classes, **kwargs)
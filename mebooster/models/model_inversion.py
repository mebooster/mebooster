from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# class Inversion_feature(nn.Module):
#     def __init__(self, nc, ngf, nz, truncation, c):
#         super(Inversion_feature, self).__init__()
#
#         self.nc = nc #3 channel
#         self.ngf = ngf #media variable
#         self.nz = nz #input feature
#         self.truncation = truncation #truncation is the number of remain input must has
#         self.c = c #50.
#
#         # self.decoder = nn.Sequential(
#         #     # input is Z
#         #     nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0),
#         #     nn.BatchNorm2d(ngf * 8),
#         #     nn.Tanh(),
#         #     # state size. (ngf*8) x 4 x 4
#         #     nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
#         #     nn.BatchNorm2d(ngf * 4),
#         #     nn.Tanh(),
#         #     # state size. (ngf*4) x 8 x 8
#         #     nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
#         #     nn.BatchNorm2d(ngf * 2),
#         #     nn.Tanh(),
#         #     # state size. (ngf*2) x 16 x 16
#         #     nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),#in_channels,out_channels,kernel_size,stride,padding
#         #     nn.BatchNorm2d(ngf),
#         #     nn.Tanh(),
#         #     # # state size. (ngf) x 32 x 32
#         #     nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
#         #     nn.Sigmoid()
#         #     # state size. (nc) x 64 x 64
#         # )
#         self.decoder = nn.Sequential(
#             # input is Z
#             nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0),
#             nn.BatchNorm2d(ngf * 4),
#             nn.Tanh(),
#             # state size. (ngf*8) x 4 x 4
#             nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
#             nn.BatchNorm2d(ngf * 2),
#             nn.Tanh(),
#             # state size. (ngf*4) x 8 x 8
#             nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
#             nn.BatchNorm2d(ngf),
#             nn.Tanh(),
#             # state size. (ngf*2) x 16 x 16
#             # nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),  # in_channels,out_channels,kernel_size,stride,padding
#             # nn.BatchNorm2d(ngf),
#             # nn.Tanh(),
#             # # state size. (ngf) x 32 x 32
#             nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         topk, indices = torch.topk(x, self.truncation)
#         topk = torch.clamp(torch.log(topk), min=-1000) + self.c
#         topk_min = topk.min(1, keepdim=True)[0]
#         topk = topk + F.relu(-topk_min)
#         x = torch.zeros(len(x), self.nz).cuda().scatter_(1, indices, topk)
#
#         x = x.view(-1, self.nz, 1, 1)# 10, 1, 1
#         # print("x0", x.shape)
#         x = self.decoder(x)
#         # print("x1", x.shape)
#         x = x.view(-1, 3, 32, 32)#-1, 1, 64, 64
#         return x


class InversionNet(nn.Module):

    def __init__(self):
        super(InversionNet, self).__init__()

        self.linear = torch.nn.Linear(100, 1024 * 2 * 2)#1024 * 4 * 4

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1024, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=3, kernel_size=4,
                stride=2, padding=1, bias=False
            )
        )
        self.out = torch.nn.Tanh()

    def forward(self, x):
        # Project and reshape
        x = self.linear(x)
        x = x.view(x.shape[0], 1024, 2, 2)
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Apply Tanh
        return self.out(x)


class DiscriminativeNet(nn.Module):

    def __init__(self):
        super(DiscriminativeNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out = nn.Sequential(
            nn.Linear(1024 * 2 * 2, 1),#1024 * 4 * 4
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Flatten and apply sigmoid
        x = x.view(-1, 1024 * 2 * 2)#1024 * 4 * 4
        x = self.out(x)
        return x

# Noise
def noise(size):
    n = Variable(torch.randn(size, 100)) #10 we don't need much,just try
    if torch.cuda.is_available(): return n.cuda()
    return n
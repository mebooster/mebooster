import argparse
import torch
from sklearn.linear_model import ridge_regression

from torch import autograd
from torch.utils.data import DataLoader, Subset

import datasets
import zoo
from blackbox import Blackbox
import config as cfg
from gaussian_train import soft_cross_entropy
from gmm import GaussianMixture
import numpy as np
from no_tenfact import no_tenfact
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from sympy import *
from math import pi

from datetime import datetime

import matplotlib.pyplot as plt
import os.path as osp

def get_teacher_model(device):
    blackbox_dir = cfg.VICTIM_DIR
    blackbox, num_classes = Blackbox.from_modeldir_split(blackbox_dir, device)
    blackbox.eval()
    return blackbox

def get_student_model(out_path, device):
    blackbox, num_classes = Blackbox.from_modeldir_split(out_path, device)
    blackbox.eval()
    return blackbox

def mnist_first(trainset0, student_model, N_query, batch_size, device):
    trainset = Subset(trainset0, np.asarray(range(N_query)))
    # test_dataset = dataset(train=False, transform=test_transform)  # , download=True

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    X_train = torch.zeros([len(trainset), 1, 28, 28]).to(device)
    y = torch.zeros([len(trainset), 10]).to(device)
    print(len(trainset))
    for idx, (data, _) in enumerate(train_loader):
        print("idx,", idx)
        X_train[idx * batch_size:min((idx + 1) * batch_size, len(trainset)), :] = data#student_model.midx(data.to(device))
        y[idx * batch_size:min((idx + 1) * batch_size, len(trainset)), :] = student_model(data.to(device))
        torch.cuda.empty_cache()

    torch.save(X_train, '.\\data\\x_train_{}.pt'.format(N_query))
    torch.save(y, ".\\data\\y_train_{}.pt".format(N_query))
    return X_train

def mnist_rand20k(trainset0, student_model, N_query, batch_size, device):
    random_idxs = np.random.choice(list(range(60000)), replace=False, size=N_query)
    trainset = Subset(trainset0, random_idxs)
    # test_dataset = dataset(train=False, transform=test_transform)  # , download=True

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    X_train = torch.zeros([len(trainset), 10]).to(device)
    y = torch.zeros([len(trainset), 10]).to(device)
    print(len(trainset))
    for idx, (data, _) in enumerate(train_loader):
        print("idx,", idx)
        X_train[idx * batch_size:min((idx + 1) * batch_size, len(trainset)), :] = student_model.midx(data.to(device))
        y[idx * batch_size:min((idx + 1) * batch_size, len(trainset)), :] = student_model(data.to(device))
        torch.cuda.empty_cache()

    torch.save(X_train, '.\\data\\x_train.pt')
    torch.save(y, ".\\data\\y_train.pt")

def main():
    device = torch.device('cuda')
    N_test = 1000
    N_query = 20000 # used to calcuate the score functions
    num_classes = 10
    batch_size =32
    #power method
    total = 2000
    torch.set_printoptions(precision=16)
    channel = 1
    #width = 6
    #x_train = torch.randn([N_query, channel, width, width]) #chw
    #torch.save(x_train, 'data\\x.pt')

    #x_test = torch.randn([N_test, channel, width, width]) #chw
    #torch.save(x_test, 'data\\x_test.pt')
    # ----------- Set up dataset

    dataset_name = 'MNIST'#params['dataset']
    valid_datasets = datasets.__dict__.keys()
    print("valid_datasets:", valid_datasets)
    if dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    dataset = datasets.__dict__[dataset_name]

    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    print("modelfamily", modelfamily)

    train_transform = datasets.modelfamily_to_transforms[modelfamily]['train']
    test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    print("train_transform:", train_transform.__class__)
    # 如果是mnist/ cifar是可以直接下载的吗
    trainset = dataset(train=True, transform=train_transform)  # , download=True
    #testset = dataset(train=False, transform=test_transform)  # , download=True
    teacher = get_teacher_model('cpu')

    stride = 4#1
    kernel = 4
    width = 28
    x_train = mnist_first(trainset, teacher, 60000, batch_size, 'cpu')
    end_steps = range(kernel, width, stride)#end index, format is :end_index
    #print("end_steps", end_steps)
    #thus, steps = len(end_steps)**2, use 4 in the center
    #becase stride=1, thus, use [9:13, 9:13] [13:17]
    step_1_x = x_train[:, :, 9:(9+kernel), 9:(9+kernel)]

    step_2_x = x_train[:, :, 9:(9+kernel), (9+stride):(9+stride+kernel)]

    step_3_x = x_train[:, :, (9+stride):(9+stride+kernel), 9:(9+kernel)]

    step_4_x = x_train[:, :, (9+stride):(9+stride+kernel), (9+stride):(9+stride+kernel)]

    torch.save(step_1_x, 'data\\step_1_x.pt') #[batch_size, 1, 3, 3]
    torch.save(step_2_x, 'data\\step_2_x.pt')
    torch.save(step_3_x, 'data\\step_3_x.pt')
    torch.save(step_4_x, 'data\\step_4_x.pt')

    # #teacher model
    # teacher_model = get_teacher_model(device)  # fc1(weight, bias), fc2(weight, bias)
    # out_path = osp.join(cfg.attack_model_dir, 'tensor_train\\basis')
    # student_model = get_student_model(out_path, device)
    # print("student_model", student_model)


if __name__ == '__main__':
    main()
#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import os.path as osp
import os
from datetime import datetime
import json
from collections import defaultdict as dd

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data as torch_data
from torch.utils.data import Dataset, DataLoader, Subset

import mebooster.config as cfg
import parser_params
from blackbox import Blackbox
from converge_main import get_optimizer
from mebooster import datasets
import mebooster.utils.transforms as transform_utils
import mebooster.utils.model_scheduler as model_utils
import mebooster.utils.utils as knockoff_utils
import mebooster.models.zoo as zoo

def convert(*cfg):
    return tuple([ v.double().cuda() for v in cfg ])

class TransferSetGaussian(Dataset):
    def __init__(self, x_train, y):
        self.data = x_train
        self.targets = y

    def __getitem__(self, index):
        x, target = self.data[index, :], self.targets[index]

        return x, target

    def __len__(self):
        return len(self.data)

def create_gaussian(N_train, N_eval, num_classes):
    # x_cls1 = np.random.multivariate_normal([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                        np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    #                                        size=int(N_train / 4))
    # x_cls2 = np.random.multivariate_normal(
    #     np.asarray([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])*2,
    #     np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), size=int(N_train / 4))
    # x_cls3 = np.random.multivariate_normal(
    #     np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])*2,
    #     np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), size=int(N_train / 4))
    # x_cls4 = np.random.multivariate_normal(
    #     np.asarray([0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0])*2,
    #     np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), size=int(N_train / 4))
    x_cls1 = np.random.multivariate_normal([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), size=int(N_train / 2))
    x_cls2 = np.random.multivariate_normal(np.asarray([0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5])*2,
                                           np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), size=int(N_train / 2))
    # x_cls3 = np.random.multivariate_normal(np.asarray([0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0])*2, np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), size=int(N_train / 4))
    # x_cls4 = np.random.multivariate_normal(np.asarray([0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0])*2, np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), size=int(N_train / 4))

    # x_cls1 = np.random.multivariate_normal([0, 0, 0, 0, 0, 0], np.diag([1, 1, 1, 1, 1, 1]),
    #                                        size=int(N_train / 4))
    # x_cls2 = np.random.multivariate_normal(np.asarray([0, 0, 0, 0.5, 0.5, 0.5]) * 2,
    #                                        np.diag([1, 1, 1, 1, 1, 1]), size=int(N_train / 4))
    # x_cls3 = np.random.multivariate_normal(np.asarray([0.5, 0.5, 0.5, 0, 0, 0]) * 2,
    #                                        np.diag([1, 1, 1, 1, 1, 1]), size=int(N_train / 4))
    # x_cls4 = np.random.multivariate_normal(np.asarray([0, 0, 0.5, 0.5, 0.5, 0]) * 2,
    #                                        np.diag([1, 1, 1, 1, 1, 1]), size=int(N_train / 4))

    x = np.vstack([x_cls1, x_cls2])
    # x = np.vstack([x, x_cls3])
    # x = np.vstack([x, x_cls4])

    x_train = torch.tensor(x).float()
    y_train = torch.zeros([N_train])
    y_train[:int(N_train / 2)] = 0.
    y_train[int(N_train / 2):2 * int(N_train / 2)] = 1.

    # y_train[2 * int(N_train / 4):3 * int(N_train / 4)] = 2.
    # y_train[3 * int(N_train / 4):4 * int(N_train / 4)] = 3.

    # x_cls1 = np.random.multivariate_normal(np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    #                                        np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), size=int(N_eval / 4))
    # x_cls2 = np.random.multivariate_normal(np.asarray([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])*2,
    #                                        np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), size=int(N_eval / 4))
    # x_cls3 = np.random.multivariate_normal(np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5 , 0.5, 0.5, 0.5, 0.5, 0.5])*2,
    #                                        np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), size=int(N_eval / 4))
    # x_cls4 = np.random.multivariate_normal(np.asarray([0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0])*2,
    #                                        np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), size=int(N_eval / 4))

    x_cls1 = np.random.multivariate_normal([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), size=int(N_eval / 2))
    x_cls2 = np.random.multivariate_normal(np.asarray([0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5])*2, np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), size=int(N_eval / 2))
    # x_cls3 = np.random.multivariate_normal(np.asarray([0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0])*2, np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), size=int(N_eval / 4))
    # x_cls4 = np.random.multivariate_normal(np.asarray([0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0])*2, np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), size=int(N_eval / 4))

    # x_cls1 = np.random.multivariate_normal([0, 0, 0, 0, 0, 0], np.diag([1, 1, 1, 1, 1, 1]),
    #                                        size=int(N_eval / 4))
    # x_cls2 = np.random.multivariate_normal(np.asarray([0, 0, 0, 0.5, 0.5, 0.5]) * 2,
    #                                        np.diag([1, 1, 1, 1, 1, 1]), size=int(N_eval / 4))
    # x_cls3 = np.random.multivariate_normal(np.asarray([0.5, 0.5, 0.5, 0, 0, 0]) * 2,
    #                                        np.diag([1, 1, 1, 1, 1, 1]), size=int(N_eval / 4))
    # x_cls4 = np.random.multivariate_normal(np.asarray([0, 0, 0.5, 0.5, 0.5, 0]) * 2,
    #                                        np.diag([1, 1, 1, 1, 1, 1]), size=int(N_eval / 4))

    x = np.vstack([x_cls1, x_cls2])
    # x = np.vstack([x, x_cls3])
    # x = np.vstack([x, x_cls4])
    x_eval = torch.tensor(x).float()
    y_eval = torch.zeros([N_eval])
    y_eval[:int(N_eval / 2)] = 0.
    y_eval[int(N_eval / 2):2 * int(N_eval / 2)] = 1.
    # y_eval[2 * int(N_eval / 4):3 * int(N_eval / 4)] = 2.
    # y_eval[3 * int(N_eval / 4):4 * int(N_eval / 4)] = 3.
    return x_train, y_train.long(), x_eval, y_eval.long()

def soft_cross_entropy(pred, soft_targets, weights=None):
    if weights is not None:
        # print("weights is not None")
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1) * weights, 1))
    else:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1))

def main():
    parser = argparse.ArgumentParser(description='Train a model')
    #arguments, not required now
    parser.add_argument('--dataset', metavar='DS_NAME', type=str, help='Dataset name', default='MNIST') #CIFAR10
    parser.add_argument('--model_arch', metavar='MODEL_ARCH', type=str, help='Model name', default='gaussian_cnn') #gaussian_nn snet
    # Optional arguments
    parser.add_argument('-o', '--out_path', metavar='PATH', type=str, help='Output path for model',
                        default=cfg.VICTIM_DIR)
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('--num_classes', metavar='D', type=int, default=2)
    parser.add_argument('-b', '--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for diff_training (default: 64)')#cfg.batch_size
    parser.add_argument('-e', '--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.1)') #0.1
    # parser.add_argument("--l2_reg_lambda", default=0.001, help="L2 regularization lambda")
    # parser.add_argument("--dropout_keep_prob", default=0.5, help="Dropout keep probability")
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging diff_training status')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--lr-step', type=int, default=30, metavar='N',
                        help='Step sizes for LR')
    parser.add_argument('--lr-gamma', type=float, default=0.1, metavar='N',
                        help='LR Decay Rate')
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('--train_subset', type=int, help='Use a subset of train set', default=None)
    parser.add_argument('--pretrained', type=str, help='Use pretrained network', default=None) #None cfg.PRETRAIN_DIR
    parser.add_argument('--weighted-loss', action='store_true', help='Use a weighted loss', default=None)
    parser.add_argument('--work_mode', action='store_true', default='victim_train')
    args = parser.parse_args()
    params = vars(args)

    # torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ----------- Set up dataset
    # data_std = 1.5
    # X_eval = torch.randn(10000, 30) * data_std #number, dim, data_std
    # X_train = torch.randn(10000, 30) * data_std
    # X_train, X_eval = convert(X_train, X_eval)

    # ----------- Set up model
    model_name = params['model_arch']
    pretrained = params['pretrained']
    num_classes = 2#params['num_classes']

    model = zoo.get_net(model_name, 'mnist', pretrained, num_classes=num_classes)
    print("model", model)
    model = model.to(device)

    model_parser = argparse.ArgumentParser(description='PyTorch ImageNet Testing')
    model_args = parser_params.add_parser_params(model_parser)

    # ----------- Train
    out_path = params['out_path']
    # model_utils.train_model(model=model, ori_model=None, trainset=trainset, testset=testset, device=device, args=model_args, **params)

    # x_train, y_train, x_eval, y_eval = create_gaussian(N_train, N_eval, params["num_classes"])
    #x_train = torch.randn(N_train, d)
    #y_train = (torch.randn(N_train)>0).long()#.float()

    #x_eval = torch.randn(N_eval, d)
    #y_eval = (torch.randn(N_eval) > 0).long()#.float()

    x = torch.load('data\\x.pt')
    N_train = x.shape[0]
    y = (torch.randn(N_train) > 0).long()
    trainset = TransferSetGaussian(x, y)

    x_eval = torch.load('data\\x_test.pt')
    N_eval = x.shape[0]
    y_eval = (torch.randn(N_eval) > 0).long()
    test_dataset = TransferSetGaussian(x_eval, y_eval)

    # dataset_name = params['dataset']
    # valid_datasets = datasets.__dict__.keys()
    # print("valid_datasets:", valid_datasets)
    # if dataset_name not in valid_datasets:
    #     raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    # dataset = datasets.__dict__[dataset_name]
    #
    # modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    # print("modelfamily", modelfamily)

    # train_transform = datasets.modelfamily_to_transforms[modelfamily]['train']
    # test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    # trainset = dataset(train=True, transform=train_transform)  # , download=True
    # test_dataset = dataset(train=False, transform=test_transform)  # , download=True

    optimizer = get_optimizer(model.parameters(), 'sgd', lr=0.01, momentum=0.5)
    # trainset = TransferSetGaussian(x_train, y_train)
    train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, pin_memory=True)

    # test_dataset = TransferSetGaussian(x_eval, y_eval)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, pin_memory=True)

    criterion_train = nn.CrossEntropyLoss(reduction='mean')

    epoch_size = len(train_loader.dataset)
    print("epoch_size", epoch_size)
    log_interval = 100

    if not osp.exists(out_path):
        knockoff_utils.create_dir(out_path)
    model_out_path = osp.join(out_path, 'checkpoint.pth.tar')
    state = {
        'epoch': 100,
        'arch': model.__class__,
        'state_dict': model.state_dict(),
        'best_acc': 100,
        'optimizer': None,
        'created_on': str(datetime.now()),
    }
    torch.save(state, model_out_path)
    small_err = 10
    best_acc=0.0
    for epoch in range(0, params['epochs']):
        model.train()
        train_loss = 0.
        err = 0
        total = 0
        acc_item = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            # inputs.requires_grad = True

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = torch.softmax(outputs, dim=1)
            # print("inputs.shape", inputs.shape)
            # print("targets.shape", targets.shape)
            loss = criterion_train(outputs, targets)
            # loss = torch.sum((1-targets)*torch.log(torch.abs(outputs)+1e-16)+targets*torch.log(torch.abs(1-outputs)+1e-16))
            # print("loss", loss.shape)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total += targets.size(0)

            _, predicted = model(inputs).max(1)
            # _, gt_label = targets.max(1) # for guassian
            gt_label=targets #for mnist
            acc_item += predicted.eq(gt_label).sum().item()

            prog = total / epoch_size
            exact_epoch = epoch + prog
            acc = acc_item/ total
            train_loss_batch = train_loss / total

            if (batch_idx + 1) % log_interval == 0:
                print('[Train] Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.4f} ({}/{})'.format(
                    exact_epoch, batch_idx * len(inputs), len(train_loader.dataset),
                                 100. * batch_idx / len(train_loader),
                    loss.item(), acc*100., acc_item, total))
        #test
        model.eval()
        test_acc_item = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            _, predicted = model(inputs).max(1)
            # _, gt_label = targets.max(1)
            gt_label = targets
            test_acc_item += predicted.eq(gt_label).sum().item()

            total += targets.size(0)
        acc = test_acc_item / total
        print("test_acc:", acc*100.)
        if acc >= best_acc:
            # print("small_err,", small_err)
            best_acc = acc
            state = {
                'epoch': 100,
                'arch': model.__class__,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': None,
                'created_on': str(datetime.now()),
            }
            torch.save(state, model_out_path)

    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(out_path, 'params.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)

if __name__ == '__main__':
    main()

#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import copy
import os.path as osp
import os
import time
from datetime import datetime
from collections import defaultdict as dd

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as torch_models

import mebooster.config as cfg
import mebooster.utils.utils as knockoff_utils

import lr_scheduler
import torch.autograd as autograd

import zoo


def get_net(model_name, n_output_classes=1000, **kwargs):
    print('=> loading model {} with arguments: {}'.format(model_name, kwargs))
    valid_models = [x for x in torch_models.__dict__.keys() if not x.startswith('__')]
    if model_name not in valid_models:
        raise ValueError('Model not found. Valid arguments = {}...'.format(valid_models))
    model = torch_models.__dict__[model_name](**kwargs)
    # Edit last FC layer to include n_output_classes
    if n_output_classes != 1000:
        if 'squeeze' in model_name:
            model.num_classes = n_output_classes
            model.classifier[1] = nn.Conv2d(512, n_output_classes, kernel_size=(1, 1))
        elif 'alexnet' in model_name:
            model.num_classes = n_output_classes
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, n_output_classes)
        elif 'vgg' in model_name:
            model.num_classes = n_output_classes
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, n_output_classes)
        elif 'dense' in model_name:
            model.num_classes = n_output_classes
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, n_output_classes)
        else:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, n_output_classes)
    return model


def soft_cross_entropy(pred, soft_targets, weights=None):
    if weights is not None:
        # print("weights is not None")
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1) * weights, 1))
    else:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1))

def train_step_vmi(model, train_loader, train_gt_loader=None, criterion=None, optimizer=None, epoch=None, device=None, log_interval=20, scheduler=None, writer=None):
    model.train()
    train_loss = 0.
    correct = 0
    total = 0
    train_loss_batch = 0
    epoch_size = len(train_loader.dataset)
    t_start = time.time()
    i=0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        scheduler(optimizer, i, epoch)
        i += 1

        optimizer.zero_grad()
        outputs = model(inputs)
        # print("outputs", outputs)
        # print("targets", targets)
        loss = criterion(outputs, targets[:, 0], targets[:, 1])
        # print("targets in diff_training step,", targets)
        if train_gt_loader is not None:
            (_, gt_labels) = train_gt_loader.__iter__().__next__()
            # print("gt_labels in diff_training step,", gt_labels)
            loss2 = criterion(outputs, gt_labels.to(device))
            loss = 0.5*loss + 0.5*loss2
        loss.backward()
        optimizer.step()
        if writer is not None:
            pass

        targets = targets[:, 0]
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if len(targets.size()) == 2:
            # Labels could be a posterior probability distribution. Use argmax as a proxy.
            target_probs, target_labels = targets.max(1)
        else:
            target_labels = targets
        correct += predicted.eq(target_labels).sum().item()

        prog = total / epoch_size
        exact_epoch = epoch + prog - 1
        acc = 100. * correct / total
        train_loss_batch = train_loss / total

        if (batch_idx + 1) % log_interval == 0:
            print('[Train] Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.1f} ({}/{})'.format(
                exact_epoch, batch_idx * len(inputs), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item(), acc, correct, total))

        if writer is not None:
            writer.add_scalar('Loss/train', loss.item(), exact_epoch)
            writer.add_scalar('Accuracy/train', acc, exact_epoch)

    t_end = time.time()
    t_epoch = int(t_end - t_start)
    acc = 100. * correct / total

    return train_loss_batch, acc

def train_step(model, train_loader, train_gt_loader=None, criterion=None, optimizer=None, epoch=None, device=None,
               log_interval=20, scheduler=None, writer=None):
    model.train()
    train_loss = 0.
    correct = 0
    total = 0
    train_loss_batch = 0
    epoch_size = len(train_loader.dataset)
    t_start = time.time()
    i=0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs.requires_grad = True

        C = inputs.size(1)
        S = inputs.size(2)
        dim = S ** 2 * C

        scheduler(optimizer, i, epoch)
        i += 1

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        if writer is not None:
            pass

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if len(targets.size()) == 2:
            # Labels could be a posterior probability distribution. Use argmax as a proxy.
            target_probs, target_labels = targets.max(1)
        else:
            target_labels = targets
        correct += predicted.eq(target_labels).sum().item()

        prog = total / epoch_size
        exact_epoch = epoch + prog - 1
        acc = 100. * correct / total
        train_loss_batch = train_loss / total

        if (batch_idx + 1) % log_interval == 0:
            print('[Train] Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.1f} ({}/{})'.format(
                exact_epoch, batch_idx * len(inputs), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item(), acc, correct, total))

        if writer is not None:
            writer.add_scalar('Loss/train', loss.item(), exact_epoch)
            writer.add_scalar('Accuracy/train', acc, exact_epoch)

    t_end = time.time()
    t_epoch = int(t_end - t_start)
    acc = 100. * correct / total

    return train_loss_batch, acc

def train_step_special(model, train_loader, train_gt_loader=None, criterion=None, optimizer=None, epoch=None, device=None,
               log_interval=20, scheduler=None, writer=None):
    # print("train_step_special")
    model.train()
    train_loss = 0.
    correct = 0
    total = 0
    train_loss_batch = 0
    epoch_size = len(train_loader.dataset)
    t_start = time.time()
    i=0
    loss_fn = torch.nn.MSELoss(reduction='mean')
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs.requires_grad = True

        scheduler(optimizer, i, epoch)
        i += 1

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs.float(), targets.float())#criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        if writer is not None:
            pass

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if len(targets.size()) == 2:
            # Labels could be a posterior probability distribution. Use argmax as a proxy.
            target_probs, target_labels = targets.max(1)
        else:
            target_labels = targets
        correct += predicted.eq(target_labels).sum().item()

        prog = total / epoch_size
        exact_epoch = epoch + prog - 1
        acc = 100. * correct / total
        train_loss_batch = train_loss / total

        if (batch_idx + 1) % log_interval == 0:
            print('[Train] Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.1f} ({}/{})'.format(
                exact_epoch, batch_idx * len(inputs), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item(), acc, correct, total))

        if writer is not None:
            writer.add_scalar('Loss/train', loss.item(), exact_epoch)
            writer.add_scalar('Accuracy/train', acc, exact_epoch)

    t_end = time.time()
    t_epoch = int(t_end - t_start)
    acc = 100. * correct / total

    return train_loss_batch, acc


def train_step_special_hidden(model, train_loader, train_gt_loader=None, criterion=None, optimizer=None, epoch=None,
                       device=None, log_interval=20, scheduler=None, layer=1, students_layer=None, writer=None):
    # print("train_step_special")
    model.train()
    train_loss = 0.
    correct = 0
    total = 0
    train_loss_batch = 0
    epoch_size = len(train_loader.dataset)
    t_start = time.time()
    i = 0
    loss_fn = torch.nn.MSELoss(reduction='mean')
    if layer == 1:
        model.features = students_layer[0]
    elif layer == 2:
        model.features = students_layer[0]
        model.fc1 = students_layer[1]

    elif layer == 3:
        model.features = students_layer[0]
        model.fc1 = students_layer[1]
        model.fc2 = students_layer[2]

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs.requires_grad = True
        scheduler(optimizer, i, epoch)
        i += 1

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs.float(), targets.float())  # criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        if writer is not None:
            pass

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if len(targets.size()) == 2:
            # Labels could be a posterior probability distribution. Use argmax as a proxy.
            target_probs, target_labels = targets.max(1)
        else:
            target_labels = targets
        correct += predicted.eq(target_labels).sum().item()

        prog = total / epoch_size
        exact_epoch = epoch + prog - 1
        acc = 100. * correct / total
        train_loss_batch = train_loss / total

        if (batch_idx + 1) % log_interval == 0:
            print('[Train] Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.1f} ({}/{})'.format(
                exact_epoch, batch_idx * len(inputs), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item(), acc, correct, total))

        # if writer is not None:
        #     writer.add_scalar('Loss/train', loss.item(), exact_epoch)
        #     writer.add_scalar('Accuracy/train', acc, exact_epoch)

        if layer == 1:
            model.fc1 = students_layer[1]
        elif layer == 2:
            model.fc1 = students_layer[1]
            model.fc2 = students_layer[2]

    t_end = time.time()
    t_epoch = int(t_end - t_start)
    acc = 100. * correct / total

    return train_loss_batch, acc

def test_step(model, test_loader, criterion, device, epoch=0., blackbox=None, silent=False, writer=None):
    model.eval()
    test_loss = 0.
    correct = 0
    correct_top5 = 0
    total = 0
    fid_num = 0
    fid_num5 = 0

    t_start = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # nclasses = outputs.size(1)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            _, pred5 = outputs.topk(5, 1, True, True)
            pred5 = pred5.t()
            correct5 = pred5.eq(targets.view(1, -1).expand_as(pred5))
            correct_top5 += correct5[:5].reshape(-1).float().sum(0, keepdim=True)  # view --> reshape

            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()
            if blackbox is not None:
                truel = blackbox(inputs)
                _, true_label =truel.max(1)
                fid_num5_t = pred5.eq(true_label.view(1, -1).expand_as(pred5))
                fid_num5 += fid_num5_t[:5].reshape(-1).float().sum(0, keepdim=True)
                fid_num += predicted.eq(true_label).sum().item()

    t_end = time.time()
    t_epoch = int(t_end - t_start)

    acc = 100. * correct / total
    fidelity = 100. * fid_num / total
    fidelity5 = 100. * fid_num5 / total
    test_loss /= total
    acc5 = 100. * correct_top5 / total

    if blackbox is not None:
        fidelity5 = fidelity5.cpu().numpy()[0]


    if not silent:
        print('[Test]  Epoch: {}\tLoss: {:.6f}\tAcc: {:.1f}% ({}/{})\t Fidelity:{}'.format(epoch, test_loss, acc,
                                                                             correct, total, fidelity))

    if writer is not None:
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', acc, epoch)
        writer.add_scalar('Fidelity/test', fidelity, epoch)

    return test_loss, acc, acc5.cpu().numpy()[0], fidelity, fidelity5


def test_step2(model, thir_model, four_model, fif_model, six_model, test_loader, criterion, device, epoch=0.,
               blackbox=None, silent=False, writer=None):
    # model.eval()
    # thir_model.eval()
    # four_model.eval()

    test_loss = 0.
    correct = 0
    total = 0
    fid_num = 0
    t_start = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # nclasses = outputs.size(1)
            if thir_model is not None:
                outputs2 = thir_model(inputs)
            else:
                outputs2 = torch.zeros([10]).to(device)
            if four_model is not None:
                outputs3 = four_model(inputs)
            else:
                outputs3 = torch.zeros([10]).to(device)
            if fif_model is not None:
                outputs4 = fif_model(inputs)
            else:
                outputs4 = torch.zeros([10]).to(device)
            if six_model is not None:
                outputs5 = six_model(inputs)
            else:
                outputs5 = torch.zeros([10]).to(device)
            test_loss += loss.item()
            _, predicted = (outputs + outputs2 + outputs3 + outputs4 +outputs5).max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if blackbox is not None:
                truel = blackbox(inputs)
                _, true_label =truel.max(1)
                fid_num += predicted.eq(true_label).sum().item()

    t_end = time.time()
    t_epoch = int(t_end - t_start)

    acc = 100. * correct / total
    fidelity = 100. * fid_num / total
    test_loss /= total

    if not silent:
        print('[Test]  Epoch: {}\tLoss: {:.6f}\tAcc: {:.1f}% ({}/{})\t Fidelity:{}'.format(epoch, test_loss, acc,
                                                                             correct, total, fidelity))

    if writer is not None:
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', acc, epoch)

    return test_loss, acc, fidelity

def test_model(blackbox=None, blackbox2=None, blackbox3=None, blackbox4=None, blackbox5=None, blackbox6=None,
               batch_size=10, testset=None,
               num_workers=10, criterion_test=None, device=None,
               epoch=100, **kwangs):
    weight = None
    if testset is not None:
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        test_loader = None
    if criterion_test is None:
        criterion_test = nn.CrossEntropyLoss(reduction='mean', weight=weight)

    if test_loader is not None:
        test_loss, test_acc, test_fidelity = test_step2(model=blackbox2, thir_model=blackbox3, four_model=blackbox4,
                                                        fif_model=blackbox5, six_model=blackbox6, test_loader=test_loader,
                                                       device=device, epoch=epoch, criterion=criterion_test,
                                                       blackbox=blackbox)


def train_model(model, ori_model, layer, trainset, trainset_gt=None, out_path=None, blackbox=None, batch_size=64, criterion_train=None, criterion_test=None, testset=None,
                device=None, num_workers=10, lr=0.1, momentum=0.5, lr_step=30, lr_gamma=0.1, resume=None,
                epochs=100, log_interval=10, weighted_loss=False, checkpoint_suffix='', optimizer=None, scheduler=None,
                writer=None, s_m=None, args=None, imp_vic_mem=False, work_mode='model_extraction', **kwargs):

    param_groups = model.parameters() if args.is_wd_all else lr_scheduler.get_parameter_groups(model)
    # if args.optimizer == 'SGD':
    print("INFO:PyTorch: using SGD optimizer.")
    #change the optimizer directly
    optimizer = torch.optim.SGD(param_groups,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True
                                )
    print('train_model_function')
    if device is None:
        device = torch.device('cuda')
    if not osp.exists(out_path):
        knockoff_utils.create_dir(out_path)
    run_id = str(datetime.now())

    # Data loaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    if trainset_gt is not None:
        train_gt_loader = DataLoader(trainset_gt, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        train_gt_loader = None
    if testset is not None:
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        test_loader = None

    # learning rate scheduler
    scheduler = lr_scheduler.lr_scheduler(mode=args.lr_mode,
                                          init_lr=args.lr,
                                          num_epochs=args.epochs,
                                          iters_per_epoch=len(train_loader),
                                          lr_milestones=args.lr_milestones,
                                          lr_step_multiplier=args.lr_step_multiplier,
                                          slow_start_epochs=args.slow_start_epochs,
                                          slow_start_lr=args.slow_start_lr,
                                          end_lr=args.end_lr,
                                          multiplier=args.lr_multiplier,
                                          decay_factor=args.decay_factor,
                                          decay_epochs=args.decay_epochs,
                                          staircase=True
                                          )

    #How
    if weighted_loss:
        if not isinstance(trainset.samples[0][1], int):
            print('Labels in trainset is of type: {}. Expected: {}.'.format(type(trainset.samples[0][1]), int))

        class_to_count = dd(int)
        for _, y in trainset.samples:
            class_to_count[y] += 1
        class_sample_count = [class_to_count[c] for c, cname in enumerate(trainset.classes)]
        print('=> counts per class: ', class_sample_count)
        weight = np.min(class_sample_count) / torch.Tensor(class_sample_count)
        weight = weight.to(device)
        print('=> using weights: ', weight)
    else:
        weight = None

    # Optimizer
    if criterion_train is None:
        criterion_train = nn.CrossEntropyLoss(reduction='mean', weight=weight)
    if criterion_test is None:
        criterion_test = nn.CrossEntropyLoss(reduction='mean', weight=weight)
    # if optimizer is None:
    #     optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    # if scheduler is None:
    #     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    start_epoch = 1
    best_train_acc, train_acc = -1., -1.
    best_test_acc, test_acc, test_loss, best_fidelity= -1., -1., -1., -1.
    # Resume if required
    if resume is not None:
        model_path = resume
        if osp.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint['epoch']
            best_test_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    # Initialize logging
    # log_path = osp.join(out_path, 'checkpoint_{}.log.tsv'.format(checkpoint_suffix))
    log_path = osp.join(out_path, '{}.log.tsv'.format(s_m))
    if not osp.exists(log_path):
        with open(log_path, 'w') as wf:
            # columns = ['run_id', 'epoch', 'split', 'loss', 'accuracy', 'best_accuracy']
            columns = ['run_id', 'loss', 'epochs', 'query_number', 'training_acc', 'best_training_acc', 'test_acc@1', 'test_acc@5', 'fidelity@1', 'fidelity@5']
            wf.write('\t'.join(columns) + '\n')
    # with open(log_path, 'a') as wf:
    #     columns = [s_m, "","","","",""]
    #     wf.write('\t'.join(columns) + '\n')

    # model_out_path = osp.join(out_path, '{}_50000.pth.tar'.format(s_m))
    model_out_path = osp.join(out_path, 'checkpoint{}.pth.tar'.format(checkpoint_suffix))
    if not cfg.layer_by_layer:
        for epoch in range(start_epoch, epochs + 1):  # 1，101
            # if imp_vic_mem:
            #     train_loss, train_acc = train_step_vmi(model, train_loader, train_gt_loader, criterion_train, optimizer, epoch,
            #                                        device, log_interval,
            #                                        scheduler=scheduler)  # log_interval=log_interval
            # else:
            if work_mode == 'victim_train':
                train_loss, train_acc = train_step(model, train_loader, train_gt_loader, criterion_train, optimizer,
                                                   epoch, device, log_interval,
                                                   scheduler=scheduler)  # log_interval=log_interval
            else:
                train_loss, train_acc = train_step_special(model, train_loader, train_gt_loader, criterion_train,
                                                           optimizer, epoch,
                                                           device, log_interval,
                                                           scheduler=scheduler)  # log_interval=log_interval
            # scheduler.step(epoch)
            best_train_acc = max(best_train_acc, train_acc)

            if True:  # (epoch+10) >=epochs:
                if test_loader is not None:
                    test_loss, test_acc, test_acc5, test_fidelity, test_fidelity5 = test_step(model, test_loader,
                                                                                              criterion_test, device,
                                                                                              epoch=epoch,
                                                                                              blackbox=blackbox)

                    if work_mode == 'victim_train':
                        is_best = (best_test_acc < test_acc)
                    else:
                        is_best = (best_fidelity < test_fidelity)
                    if is_best:
                        best_test_acc = test_acc
                        best_fidelity = test_fidelity
                        best_test_acc5 = test_acc5
                        best_fidelity5 = test_fidelity5

                        # Checkpoint
                        # if test_acc >= best_test_acc:
                        state = {
                            'epoch': epoch,
                            'arch': model.__class__,
                            'state_dict': model.state_dict(),
                            'best_acc': test_acc,
                            'optimizer': optimizer.state_dict(),
                            'created_on': str(datetime.now()),
                        }
                        torch.save(state, model_out_path)
            # Log
            if epoch % 10 == 0:
                with open(log_path, 'a') as af:
                    train_cols = [run_id, train_loss, epoch, len(trainset), train_acc, best_train_acc, test_acc,
                                  best_test_acc, best_test_acc5,
                                  test_fidelity, best_fidelity, best_fidelity5]
                    af.write('\t'.join([str(c) for c in train_cols]) + '\n')
                    # test_cols = [run_id, epoch, 'test', test_loss, test_acc, best_test_acc, test_fidelity]
                    # af.write('\t'.join([str(c) for c in test_cols]) + '\n')
    # columns = ['run_id', 'loss', 'query_number', 'training_acc', 'test_acc', 'fidelity']

    # train another layer
    if cfg.layer_by_layer:
        #train other layers

        print("train-[layer]:", layer + 1)

        students_layer = dict()
        students_layer[0] = ori_model.features
        students_layer[1] = ori_model.fc1
        students_layer[2] = ori_model.fc2
        # model = copy.deepcopy(ori_model)

        for epoch in range(start_epoch, epochs + 1):  # 1，101
            train_loss, train_acc = train_step_special_hidden(model, train_loader, train_gt_loader, criterion_train,
                                                              optimizer, epoch,
                                                              device, log_interval,
                                                              scheduler=scheduler, layer=layer,
                                                              students_layer=students_layer)  # log_interval=log_interval
            # scheduler.step(epoch)
            best_train_acc = max(best_train_acc, train_acc)

            if True:  # (epoch+10) >=epochs:
                if test_loader is not None:
                    test_loss, test_acc, test_acc5, test_fidelity, test_fidelity5 = test_step(model, test_loader,
                                                                                              criterion_test,
                                                                                              device, epoch=epoch,
                                                                                              blackbox=blackbox)
                    if work_mode == 'victim_train':
                        is_best = (best_test_acc < test_acc)
                    else:
                        is_best = (best_fidelity < test_fidelity)
                    if is_best:
                        best_test_acc = test_acc
                        best_fidelity = test_fidelity
                        best_test_acc5 = test_acc5
                        best_fidelity5 = test_fidelity5

                        # Checkpoint
                        # if test_acc >= best_test_acc:
                        state = {
                            'epoch': epoch,
                            'arch': model.__class__,
                            'state_dict': model.state_dict(),
                            'best_acc': test_acc,
                            'optimizer': optimizer.state_dict(),
                            'created_on': str(datetime.now()),
                        }
                        torch.save(state, model_out_path)
            # Log
            if epoch % 10 == 0:
                with open(log_path, 'a') as af:
                    train_cols = [run_id, train_loss, epoch, len(trainset), train_acc, best_train_acc, test_acc,
                                  best_test_acc, best_test_acc5,
                                  test_fidelity, best_fidelity, best_fidelity5]
                    af.write('\t'.join([str(c) for c in train_cols]) + '\n')

    with open(log_path, 'a') as af:
        train_cols = [run_id, train_loss, epoch, len(trainset), train_acc, best_train_acc, test_acc, best_test_acc, best_test_acc5,
                      test_fidelity, best_fidelity, best_fidelity5]
        af.write('\t'.join([str(c) for c in train_cols]) + '\n')
        # test_cols = [run_id, epoch, 'test', test_loss, test_acc, best_test_acc, test_fidelity]
        # af.write('\t'.join([str(c) for c in test_cols]) + '\n')

    return model

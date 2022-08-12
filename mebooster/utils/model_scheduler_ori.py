#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
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
    #一个训练过程
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

def train_step(model, train_loader, train_gt_loader=None, criterion=None, optimizer=None, epoch=None, device=None, log_interval=20, scheduler=None, writer=None):
    model.train()

    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.conv2.parameters():
        param.requires_grad = False

    train_loss = 0.
    correct = 0
    total = 0
    train_loss_batch = 0
    epoch_size = len(train_loader.dataset)
    t_start = time.time()
    #一个训练过程
    i=0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        scheduler(optimizer, i, epoch)
        i += 1

        optimizer.zero_grad()
        outputs = model(inputs)
        # print("outputs", outputs)
        # print("targets", targets)
        loss = criterion(outputs, targets)
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


def train_model(model, trainset, trainset_gt=None, out_path=None, blackbox=None, batch_size=64, criterion_train=None, criterion_test=None, testset=None,
                device=None, num_workers=10, lr=0.1, momentum=0.5, lr_step=30, lr_gamma=0.1, resume=None,
                epochs=100, log_interval=10, weighted_loss=False, checkpoint_suffix='', optimizer=None, scheduler=None,
                writer=None, s_m=None, args=None, imp_vic_mem=False, work_mode='model_extraction', mode='over', **kwargs):
    #change optimizer
    #transfer learning

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
    if weighted_loss:#loss 是有权重的
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
    # Resume if required 从某个模型继续训练
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
    model_out_path = osp.join(out_path, 'checkpoint_{}_{}.pth.tar'.format(checkpoint_suffix, mode))
    for epoch in range(start_epoch, epochs + 1): #1，101
        #在这里就跑完了一个epoch
        if imp_vic_mem:
            train_loss, train_acc = train_step_vmi(model, train_loader, train_gt_loader, criterion_train, optimizer, epoch,
                                               device, log_interval,
                                               scheduler=scheduler)  # log_interval=log_interval
        else:
            train_loss, train_acc = train_step(model, train_loader, train_gt_loader, criterion_train, optimizer, epoch, device, log_interval,
                                           scheduler=scheduler)#log_interval=log_interval
        # scheduler.step(epoch)
        best_train_acc = max(best_train_acc, train_acc)

        if True:#(epoch+10) >=epochs:
            if test_loader is not None:
                test_loss, test_acc, test_acc5, test_fidelity, test_fidelity5 = test_step(model, test_loader, criterion_test, device, epoch=epoch,
                                                               blackbox=blackbox)
                if work_mode=='victim_train':
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
                train_cols = [run_id, train_loss, epoch, len(trainset), train_acc, best_train_acc, test_acc, best_test_acc, best_test_acc5,
                      test_fidelity, best_fidelity, best_fidelity5]
                af.write('\t'.join([str(c) for c in train_cols]) + '\n')
                # test_cols = [run_id, epoch, 'test', test_loss, test_acc, best_test_acc, test_fidelity]
                # af.write('\t'.join([str(c) for c in test_cols]) + '\n')
    #columns = ['run_id', 'loss', 'query_number', 'training_acc', 'test_acc', 'fidelity']
    with open(log_path, 'a') as af:
        train_cols = [run_id, train_loss, epoch, len(trainset), train_acc, best_train_acc, test_acc, best_test_acc, best_test_acc5,
                      test_fidelity, best_fidelity, best_fidelity5]
        af.write('\t'.join([str(c) for c in train_cols]) + '\n')
        # test_cols = [run_id, epoch, 'test', test_loss, test_acc, best_test_acc, test_fidelity]
        # af.write('\t'.join([str(c) for c in test_cols]) + '\n')

    return model

def act2corrMat(src, dst):
    ''' src[:, k], with k < K1
        dst[:, k'], with k' < K2
        output correlation score[K1, K2]
    '''
    # K_src by K_dst
    if len(src.size()) == 3 and len(dst.size()) == 3:
        src = src.permute(0, 2, 1).contiguous().view(src.size(0) * src.size(2), -1)
        dst = dst.permute(0, 2, 1).contiguous().view(dst.size(0) * dst.size(2), -1)

    # conv activations.
    elif len(src.size()) == 4 and len(dst.size()) == 4:
        src = src.permute(0, 2, 3, 1).contiguous().view(src.size(0) * src.size(2) * src.size(3), -1)
        dst = dst.permute(0, 2, 3, 1).contiguous().view(dst.size(0) * dst.size(2) * dst.size(3), -1)

    # Substract mean.
    src = src - src.mean(0, keepdim=True)
    dst = dst - dst.mean(0, keepdim=True)

    inner_prod = torch.mm(src.t(), dst)
    src_inv_norm = src.pow(2).sum(0).add_(1e-10).rsqrt().view(-1, 1)
    dst_inv_norm = dst.pow(2).sum(0).add_(1e-10).rsqrt().view(1, -1)

    return inner_prod * src_inv_norm * dst_inv_norm

def rec_train_step(model, train_loader, train_gt_loader=None, criterion=None, optimizer=None, epoch=None, device=None, log_interval=20, scheduler=None, writer=None):
    # model.train()
    train_loss = 0.
    correct = 0
    total = 0
    train_loss_batch = 0
    epoch_size = len(train_loader.dataset)
    t_start = time.time()
    #一个训练过程
    i=0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # print("batch_idx", batch_idx)
        inputs, targets = inputs.to(device), targets.to(device)
        scheduler(optimizer, i, epoch)
        i += 1

        optimizer.zero_grad()
        outputs = model(inputs)
        # print("outputs", outputs)
        # print("targets", targets)
        loss = criterion(outputs, targets)
        # print("targets in diff_training step,", targets)
        # if train_gt_loader is not None:
        #     (_, gt_labels) = train_gt_loader.__iter__().__next__()
        #     # print("gt_labels in diff_training step,", gt_labels)
        #     loss2 = criterion(outputs, gt_labels.to(device))
        #     loss = 0.5*loss + 0.5*loss2

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

#for resnet18
def rec_train_model(model, trainset, trainset_gt=None, out_path=None, blackbox=None, batch_size=64, criterion_train=None, criterion_test=None, testset=None,
                device=None, num_workers=0, lr=0.1, momentum=0.5, lr_step=30, lr_gamma=0.1, resume=None,
                epochs=100, log_interval=10, weighted_loss=False, checkpoint_suffix='', optimizer=None, scheduler=None,
                writer=None, s_m=None, args=None, imp_vic_mem=False, work_mode='model_extraction', mode='over', **kwargs):
    #change optimizer
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
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True) #, num_workers=num_workers
    print("trainset,", len(trainset))
    # if trainset_gt is not None:
    #     train_gt_loader = DataLoader(trainset_gt, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    # else:
    #     train_gt_loader = None
    train_gt_loader = None
    if testset is not None:
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True) #num_workers=num_workers,
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
                                          staircase=True)

    #How
    if weighted_loss:#loss 是有权重的
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
    # Resume if required 从某个模型继续训练
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
    model.train()
    for layer_index in range(0, 4):
        model_out_path = osp.join(out_path, 'checkpoint_{}_{}_{}.pth.tar'.format(checkpoint_suffix, mode, layer_index))

        #resnet18
        # if layer_index == 1:
        #     for param in model.conv1.parameters():
        #         param.requires_grad = False
        # if layer_index == 2:
        #     for param in model.conv1.parameters():
        #         param.requires_grad = False
        #     for param in model.layer1[0].conv1.parameters():
        #         param.requires_grad = False
        # if layer_index == 3:
        #     for param in model.conv1.parameters():
        #         param.requires_grad = False
        #     for param in model.layer1[0].conv1.parameters():
        #         param.requires_grad = False
        #     for param in model.layer1[0].conv2.parameters():
        #         param.requires_grad = False
        # if layer_index == 4:
        #     for param in model.conv1.parameters():
        #         param.requires_grad = False
        #     for param in model.layer1[0].conv1.parameters():
        #         param.requires_grad = False
        #     for param in model.layer1[0].conv2.parameters():
        #         param.requires_grad = False
        #     for param in model.layer1[1].conv1.parameters():
        #         param.requires_grad = False

        for epoch in range(start_epoch, epochs + 1):  # 1，101
            # if layer_index == 0:
            #     accumu_grad = model.conv1.weight.view(model.conv1.weight.shape[0], -1).T
            # elif layer_index == 1:
            #     accumu_grad = model.layer1[0].conv1.weight.view(model.layer1[0].conv1.weight.shape[0], -1).T
            # elif layer_index == 2:
            #     accumu_grad = model.layer1[0].conv2.weight.view(model.layer1[0].conv2.weight.shape[0], -1).T
            # elif layer_index == 3:
            #     accumu_grad = model.layer1[1].conv1.weight.view(model.layer1[1].conv1.weight.shape[0], -1).T
            # elif layer_index == 4:
            #     accumu_grad = model.layer1[1].conv2.weight.view(model.layer1[1].conv2.weight.shape[0], -1).T

            train_loss, train_acc = rec_train_step(model, train_loader, train_gt_loader, criterion_train, optimizer,
                                                   epoch, device, log_interval,
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

            # if layer_index == 0:
            #     if (torch.diag(act2corrMat(model.conv1.weight.view(model.conv1.weight.shape[0], -1).T,
            #                                accumu_grad)) > (1 - 0.1*10e-7)).sum() == model.conv1.weight.shape[0]:
            #         with open(log_path, 'a') as af:
            #             train_cols = [run_id, train_loss, epoch, len(trainset), train_acc, best_train_acc, test_acc,
            #                           best_test_acc, best_test_acc5,
            #                           test_fidelity, best_fidelity, best_fidelity5]
            #             af.write('\t'.join([str(c) for c in train_cols]) + '\n')
            #         break
            # elif layer_index == 1:
            #     if (torch.diag(act2corrMat(model.layer1[0].conv1.weight.view(model.layer1[0].conv1.weight.shape[0], -1).T,
            #                                accumu_grad)) > (1 - 0.1*10e-7)).sum() == model.layer1[0].conv1.weight.shape[0]:
            #         with open(log_path, 'a') as af:
            #             train_cols = [run_id, train_loss, epoch, len(trainset), train_acc, best_train_acc, test_acc,
            #                           best_test_acc, best_test_acc5,
            #                           test_fidelity, best_fidelity, best_fidelity5]
            #             af.write('\t'.join([str(c) for c in train_cols]) + '\n')
            #         break
            # elif layer_index == 2:
            #     if (torch.diag(act2corrMat(model.layer1[0].conv2.weight.view(model.layer1[0].conv2.weight.shape[0], -1).T,
            #                                accumu_grad)) > (1 - 0.1*10e-7)).sum() == model.layer1[0].conv2.weight.shape[0]:
            #         with open(log_path, 'a') as af:
            #             train_cols = [run_id, train_loss, epoch, len(trainset), train_acc, best_train_acc, test_acc,
            #                           best_test_acc, best_test_acc5,
            #                           test_fidelity, best_fidelity, best_fidelity5]
            #             af.write('\t'.join([str(c) for c in train_cols]) + '\n')
            #         break
            # elif layer_index == 3:
            #     if (torch.diag(act2corrMat(model.layer1[1].conv1.weight.view(model.layer1[1].conv1.weight.shape[0], -1).T,
            #                                accumu_grad)) > (1 - 0.1*10e-7)).sum() == model.layer1[1].conv1.weight.shape[0]:
            #         with open(log_path, 'a') as af:
            #             train_cols = [run_id, train_loss, epoch, len(trainset), train_acc, best_train_acc, test_acc,
            #                           best_test_acc, best_test_acc5,
            #                           test_fidelity, best_fidelity, best_fidelity5]
            #             af.write('\t'.join([str(c) for c in train_cols]) + '\n')
            #         break
            # elif layer_index == 4:
            #     if (torch.diag(act2corrMat(model.layer1[1].conv2.weight.view(model.layer1[1].conv2.weight.shape[0], -1).T,
            #                                accumu_grad)) > (1 - 0.1*10e-7)).sum() == model.layer1[1].conv2.weight.shape[0]:
            #         with open(log_path, 'a') as af:
            #             train_cols = [run_id, train_loss, epoch, len(trainset), train_acc, best_train_acc, test_acc,
            #                           best_test_acc, best_test_acc5,
            #                           test_fidelity, best_fidelity, best_fidelity5]
            #             af.write('\t'.join([str(c) for c in train_cols]) + '\n')
            #         break

    with open(log_path, 'a') as af:
        train_cols = [run_id, train_loss, epoch, len(trainset), train_acc, best_train_acc, test_acc, best_test_acc, best_test_acc5,
                      test_fidelity, best_fidelity, best_fidelity5]
        af.write('\t'.join([str(c) for c in train_cols]) + '\n')

    return model
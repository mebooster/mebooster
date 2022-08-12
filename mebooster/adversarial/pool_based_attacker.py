#!/usr/bin/python
"""In this file, i realize the shadow model MIA with no previous assumptions
"""

import argparse
import copy
import json
import math
import os
import os.path as osp
import pickle
# from datetime import datetime

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader
from torch import optim
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader

import classifier
import config as cfg
import model_scheduler_ori as model_utils
# import knockoff.utils.split_model as split_model_utils
# from bayes_attack import BayesMemberAttack
import norm
import parser_params
import splitnet
from adversarial_deepfool_sss import AdversarialDeepFoolStrategy
from autoaugment import CIFAR10Policy
from bayesian_disagreement_dropout_sss import BALDDropoutStrategy
from graph_density_sss import GraphDensitySelectionStrategy
from kcenter_sss import KCenterGreedyApproach
import datasets
import models.zoo as zoo
from victim.blackbox import Blackbox
from tqdm import tqdm
import random

from margin_sss import MarginSelectionStrategy
from random_sss import RandomSelectionStrategy
from uncertainty_sss import UncertaintySelectionStrategy
from utils.utils import clipDataTopX
import time
import datetime
from torch.nn import Parameter

class TransferSetImagePaths(ImageFolder):
    """TransferSet Dataset, for when images are stored as *paths*"""

    def __init__(self, samples, transform=None, target_transform=None):
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.target_transform = target_transform

class TransferSetImages(Dataset):
    def __init__(self, samples, transform=None, target_transform=None):
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

        self.data = [self.samples[i][0] for i in range(len(self.samples))]
        self.targets = [self.samples[i][1] for i in range(len(self.samples))]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

def samples_to_transferset(samples, budget=None, transform=None, target_transform=None):
    # Images are either stored as paths, or numpy arrays
    sample_x = samples[0][0]
    assert budget <= len(samples), 'Required {} samples > Found {} samples'.format(budget, len(samples))

    if isinstance(sample_x, str):
        return TransferSetImagePaths(samples[:budget], transform=transform, target_transform=target_transform)
    elif isinstance(sample_x, np.ndarray):
        return TransferSetImages(samples[:budget], transform=transform, target_transform=target_transform)
    else:
        raise ValueError('type(x_i) ({}) not recognized. Supported types = (str, np.ndarray)'.format(type(sample_x)))

def get_optimizer(parameters, optimizer_type, lr=0.01, momentum=0.5, **kwargs):
    assert optimizer_type in ['sgd', 'sgdm', 'adam', 'adagrad']
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(parameters, lr)
    elif optimizer_type == 'sgdm':
        optimizer = optim.SGD(parameters, lr, momentum=momentum)
    elif optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(parameters)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(parameters)
    else:
        raise ValueError('Unrecognized optimizer type')
    return optimizer

class GetAdversary(object):
    def __init__(self, blackbox, queryset, batch_size=8):
        self.blackbox = blackbox
        self.attack_model = None
        self.attack_device = None
        self.queryset = queryset

        self.n_queryset = len(self.queryset)
        self.batch_size = batch_size
        self.idx_set = set()
        self.q_idx_set = set()

        self.transferset = []  # List of tuples [(img_path, output_probs)]
        self.transfery = []
        self.pre_idxset = []
        self._restart()
        self.sampling_method = 'random'
        self.ma_method = 'unsupervised'

        self.no_training_in_initial = 0

    def _restart(self):
        np.random.seed(cfg.DEFAULT_SEED)
        torch.manual_seed(cfg.DEFAULT_SEED)
        torch.cuda.manual_seed(cfg.DEFAULT_SEED)

        self.idx_set = set(range(len(self.queryset))) #idx根据queryset进行更改的
        self.transferset = []
        self.transfery = []

    def set_attack_model(self, attack_model, device):
        # self.attack_model = Blackbox(attack_model, device, 'probs')
        self.attack_model = attack_model
        self.attack_device = device
        # self.attack_model.eval()

    # For KCenter
    def get_initial_centers(self):
        Y_vec_true = []
        print("get_initial_centers")
        assert self.attack_model is not None, "attack_model made a mistake!"
        for b in range(int(np.ceil(len(self.pre_idxset)/self.batch_size))): #不是这么回事
            # print("b = ", b)
            # print("pre_dixset = ", self.pre_idxset)
            x_idx = self.pre_idxset[(b * self.batch_size): min(((b+1) * self.batch_size), len(self.pre_idxset))]
            # print("x_idx:", x_idx)
            trX = torch.stack([self.queryset[int(i)][0] for i in x_idx]).to(self.attack_device)
            trY = self.attack_model(trX).cpu()
            Y_vec_true.append(trY)
        Y_vec_true = np.concatenate(Y_vec_true)
        # print("in get_initial_centers:,len(x_idx):", len(Y_vec_true))
        # print("Y_vec_true,", Y_vec_true.shape)
        return Y_vec_true

    def get_transferset(self, k, sampling_method='random', ma_method='unsupervised', pre_idxset=[],
                        shadow_attack_model=None, device=None, second_sss=None, it=None, initial_seed=[]):
        self.sampling_method=sampling_method
        self.ma_method=ma_method
        print("pre_idxset:", len(pre_idxset))
        start_B = 0
        end_B = k
        dt=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("start_query_strategy:", dt)
        with tqdm(total=k) as pbar:
            if self.sampling_method == 'initial_seed' or self.sampling_method == 'training_data' or\
                self.sampling_method == 'label':
                for t, B in enumerate(range(start_B, end_B, self.batch_size)):  # 1,200;步长：8
                    # print("start", B)
                    if self.sampling_method == 'training_data':
                        self.q_idx_set = set(range(60000))
                        self.q_idx_set.intersection_update(self.idx_set)
                    else:
                        self.q_idx_set = copy.copy(self.idx_set)

                    idxs = np.random.choice(list(self.q_idx_set), replace=False,
                                            size=min(self.batch_size, k-len(self.pre_idxset)))  # 8，200-目前拥有的transferset的大小。
                    print("initial_seed_idxs", idxs)
                    #这就是选出的idx了
                    for index in idxs:
                        if index >= cfg.trainingbound:
                            self.no_training_in_initial += 1

                    self.idx_set = self.idx_set - set(idxs)
                    #这里存储 选出的pre_idxset
                    self.pre_idxset = np.append(self.pre_idxset, idxs)
                    # print("idx,", idxs)
                    if len(self.idx_set) == 0:
                        print('=> Query set exhausted. Now repeating input examples.')
                        self.idx_set = set(range(len(self.queryset)))

                    x_t = torch.stack([self.queryset[i][0] for i in idxs]).to(self.blackbox.device)
                    y_t = self.blackbox(x_t).cpu()

                    #目前全部按照ChainDataset来说
                    if hasattr(self.queryset, 'samples'):
                        # Any DatasetFolder (or subclass) has this attribute # Saving image paths are space-efficient
                        img_t = [self.queryset.samples[i][0] for i in idxs]  # Image paths
                    else:
                        # Otherwise, store the image itself # But, we need to store the non-transformed version
                        # img_t = [self.queryset.data[i] for i in idxs]
                        if len(self.queryset.data) <= 3:
                            img_t, gt_label = self.queryset.getitemsinchain(idxs)
                        else:
                            img_t = [self.queryset.data[i] for i in idxs]
                            # gt_label = [self.queryset.targets[i] for i in idxs]
                        # if isinstance(self.queryset.data[0], torch.Tensor):
                        #     img_t = [x.numpy() for x in img_t]

                    for i in range(len(idxs)):
                        img_t_i = img_t[i].numpy() if isinstance(img_t[i], torch.Tensor) else img_t[i]
                        img_t_i = img_t_i.squeeze() if isinstance(img_t_i, np.ndarray) else img_t_i
                        self.transferset.append((img_t_i, y_t[i].squeeze()))
                        self.transfery.append(y_t[i].squeeze().numpy())
                    pbar.update((x_t.size(0)))
            elif self.sampling_method == 'use_default_initial':
                assert len(initial_seed) > 0, 'has no input initial seed!'
                chosed_idx = [list(self.idx_set)[int(e)] for e in initial_seed]  # 让idx_set减去这个chosed_idx；已经做出了选择
                self.idx_set = self.idx_set - set(chosed_idx)
                self.pre_idxset = np.append(self.pre_idxset, chosed_idx)
                print("self.pre_idxset:", self.pre_idxset)
                for index in chosed_idx:
                    if index >= cfg.trainingbound:
                        self.no_training_in_initial += 1
                # Query
                for b in range(int(np.ceil(len(initial_seed) / self.batch_size))):
                    # x_b = x_t[b * self.batch_size: min((1 + b) * self.batch_size, len(s))].to(self.blackbox.device)
                    c_idx = chosed_idx[b * self.batch_size: min((1 + b) * self.batch_size, len(initial_seed))]
                    x_b = torch.stack([self.queryset[i][0] for i in c_idx]).to(self.blackbox.device)
                    y_b = self.blackbox(x_b).cpu()

                    if hasattr(self.queryset, 'samples'):
                        # Any DatasetFolder (or subclass) has this attribute # Saving image paths are space-efficient
                        img_t = [self.queryset.samples[i][0] for i in c_idx]  # Image paths
                    else:
                        # Otherwise, store the image itself # But, we need to store the non-transformed version
                        if len(self.queryset.data) <= 3:
                            # print("\nwe use a mnist chain")
                            img_p, _ = self.queryset.getitemsinchain(c_idx)
                        else:
                            img_p = [self.queryset.data[i] for i in c_idx]

                    for m in range(len(c_idx)):
                        img_p_i = img_p[m].numpy() if isinstance(img_p[m], torch.Tensor) else img_p[m]
                        img_t_i = img_p_i.squeeze() if isinstance(img_p_i, np.ndarray) else img_p_i
                        self.transferset.append((img_t_i, y_b[m].squeeze()))  # self.transferset
                        self.transfery.append(y_b[m].squeeze().numpy())
                    pbar.update(x_b.size(0))

        print('self.idx_set', len(self.idx_set))
        print('self.pre_idxset', len(self.pre_idxset))
        dt1 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("start_query_strategy:", dt)
        print("start_query_strategy:", dt1)
        return self.transferset, self.pre_idxset

def Parser():
    parser = argparse.ArgumentParser(description='Train a model')
    # -----------------------Query arguments
    parser.add_argument('--victim_model_dir', metavar='PATH', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"',
                        default=cfg.VICTIM_DIR)
    parser.add_argument('--out_dir', metavar='PATH', type=str,
                        help='Destination directory to store transfer set',
                        default=cfg.transfer_set_out_dir)  # required=True,
    parser.add_argument('--queryset', metavar='TYPE', type=str, help='Adversary\'s dataset (P_A(X))',
                        default=cfg.queryset)  # required=True
    parser.add_argument('--root', metavar='DIR', type=str, help='Root directory for ImageFolder', default=None)
    parser.add_argument('--modelfamily', metavar='TYPE', type=str, help='Model family', default=None)
    parser.add_argument("-sampling_method", default=cfg.sampling_method,
                        help="sampling method")  # random, uncertainty, adversarial, kcenter and adversarial-kcenter, adaptive(reinforcement)
    parser.add_argument("-ma_method", default=cfg.ma_method,
                        help="sampling method")  # naive, shadow_model, bayse, generative_attack, unsupervised, gradient(need training data),
    parser.add_argument("-second_sss", default=None) #random, uncertainty, adversarial, kcenter
    # ----- assert in iterative
    parser.add_argument("-iterative", default=True, help="use iterative training method or not")
    parser.add_argument("-copy_one_hot", default=cfg.copy_one_hot, help="use one-hot copy or softmax")
    parser.add_argument("-initial_seed", default=cfg.initial_seed, help="intial seed")  #### None; 200
    parser.add_argument("-num_iter", default=cfg.num_iter, help="num of iterations")  #### None;
    parser.add_argument("-k", default=cfg.k,
                        help="add queries")  ## 10*100 ## None; k samples are chosen in accordance with the sampling_method

    # ----------------------Required arguments
    parser.add_argument('--attack_model_dir', metavar='DIR', type=str, help='Directory containing transferset.pickle',
                        default=cfg.attack_model_dir)
    parser.add_argument('--model_arch', metavar='MODEL_ARCH', type=str, help='Model name', default=cfg.attack_model_arch)
    parser.add_argument('--testdataset', metavar='DS_NAME', type=str, help='Name of test', default=cfg.test_dataset)
    # ----------------------Optional arguments; train dataset
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch_size', metavar='TYPE', type=int, help='Batch size of queries', default=cfg.batch_size)
    parser.add_argument('-e', '--epochs', type=int, default=cfg.epoch, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log-interval', type=int, default=cfg.log_interval, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--lr-step', type=int, default=60, metavar='N',
                        help='Step sizes for LR')
    parser.add_argument('--lr-gamma', type=float, default=0.1, metavar='N',
                        help='LR Decay Rate')
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('--pretrained', type=str, help='Use pretrained network', default=None)
    parser.add_argument('--weighted-loss', action='store_true', help='Use a weighted loss', default=False)
    # Attacker's defense
    parser.add_argument('--argmaxed', action='store_true', help='Only consider argmax labels', default=False)
    parser.add_argument('--optimizer_choice', type=str, help='Optimizer', default='sgdm',
                        choices=('sgd', 'sgdm', 'adam', 'adagrad'))

    parser.add_argument('--shadow_data', type=int, default=cfg.shadow_data)

    return parser

def over_initial_cnn(w1_dir1, w1_dir2,  w1_dir3, w1_dir4, w1_dir5, channel=3):
    #over_factor
    d, m1 = w1_dir1.shape
    w1 = w1_dir1
    w2 = w1_dir2
    w3 = w1_dir3
    w4 = w1_dir4
    w5 = w1_dir5 #d * m1
    init_w1_1 = ((w1/torch.norm(w1, dim=1).view(-1, 1)) * math.sqrt(2./d))#mag1# * w1 #d*k torch.randn([k, d]).to(device)
    init_w1_2 = (w2/torch.norm(w2, dim=1).view(-1, 1)) * math.sqrt(2./d)#mag3#torch.randn([k, d]).to(device) * math.sqrt(2/d)
    init_w1_3 = (w3 / torch.norm(w3, dim=1).view(-1, 1)) * math.sqrt(2. / d)
    init_w1_4 = (w4 / torch.norm(w4, dim=1).view(-1, 1)) * math.sqrt(2./d)#mag4
    init_w1_5 = (w5 / torch.norm(w5, dim=1).view(-1, 1)) * math.sqrt(2./d)#mag5

    d, m1 = w1_dir1.shape
    # print("init_w1_1,", init_w1_1-init_w1_2)
    # print("init_w1_2,", init_w1_2.shape)
    # init_w1_3 = torch.randn_like(w1_dir).to(device)#mag3 * w1
    # m1 * d
    init_w1 = torch.vstack((init_w1_1.T, init_w1_2.T, init_w1_3.T, init_w1_4.T, init_w1_5.T)).view(m1*5, channel,
                                                                                                   int(np.sqrt(d/channel)), int(np.sqrt(d/channel))) #init_w1_4.T, init_w1_5.T
    return init_w1

def over_initial(w1_dir1, w1_dir2,  w1_dir3, w1_dir4, w1_dir5):
    #over_factor
    d, m1 = w1_dir1.shape
    w1 = w1_dir1
    w2 = w1_dir2
    w3 = w1_dir3
    w4 = w1_dir4
    w5 = w1_dir5 #d * m1
    init_w1_1 = ((w1 / torch.norm(w1, dim=1).view(-1, 1)) * math.sqrt(2./d))#mag1# * w1 #d*k torch.randn([k, d]).to(device)
    init_w1_2 = (w2 / torch.norm(w2, dim=1).view(-1, 1)) * math.sqrt(2./d)#mag3#torch.randn([k, d]).to(device) * math.sqrt(2/d)
    init_w1_3 = (w3 / torch.norm(w3, dim=1).view(-1, 1)) * math.sqrt(2. / d)
    init_w1_4 = (w4 / torch.norm(w4, dim=1).view(-1, 1)) * math.sqrt(2./d)#mag4
    init_w1_5 = (w5 / torch.norm(w5, dim=1).view(-1, 1)) * math.sqrt(2./d)#mag5

    d, m1 = w1_dir1.shape
    # print("init_w1_1,", init_w1_1-init_w1_2)
    # print("init_w1_2,", init_w1_2.shape)
    # init_w1_3 = torch.randn_like(w1_dir).to(device)#mag3 * w1
    # m1 * d
    init_w1 = torch.vstack((init_w1_1.T, init_w1_2.T, init_w1_3.T, init_w1_4.T, init_w1_5.T)).view(m1*5, d) #init_w1_4.T, init_w1_5.T
    return init_w1

def main():
    parser = Parser()
    args = parser.parse_args()
    params = vars(args)
    # ----------- Seed, device, attack_dir
    torch.manual_seed(cfg.DEFAULT_SEED)
    np.random.seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    attack_model_dir = params['attack_model_dir']
    shadow_model_dir = cfg.shadow_model_dir
    valid_datasets = datasets.__dict__.keys()

    queryset_name = params['queryset']
    queryset_names = queryset_name.split(',')
    # print("valid_datasets: ", valid_datasets)
    for i, qname in enumerate(queryset_names):
        if qname.find("-") > -1:
            qname = qname.split("-")[0]
        if qname not in valid_datasets:  # 几大data family
            raise ValueError('Dataset not found. Valid arguments = {}, qname= {}'.format(valid_datasets, qname))
        modelfamily = datasets.dataset_to_modelfamily[qname] if params['modelfamily'] is None else params[
            'modelfamily']
        print("modelfamily,", modelfamily)
        break
    # 目前全来自一个家族MNIST
    transform_query = datasets.modelfamily_to_transforms[modelfamily]['train']

    # ----------- Set up testset
    test_dataset_name = params['testdataset']  # 用的是MNIST test
    test_modelfamily = datasets.dataset_to_modelfamily[test_dataset_name]
    test_transform = datasets.modelfamily_to_transforms[test_modelfamily]['test']
    print("test_transform:", test_transform.__dict__.keys())

    if queryset_name == 'ImageFolder':
        assert params['root'] is not None, 'argument "--root ROOT" required for ImageFolder'
        queryset = datasets.__dict__[queryset_name](root=params['root'], transform=test_transform)
    elif len(queryset_names) > 1:  # 拥有多个dataset
        qns = "ChainMNIST"
        for qn in queryset_names:
            if qn.find("CIFAR10") == 0:
                qns = "ChainCIFAR"
                break
        queryset = datasets.__dict__[qns](chain=queryset_names, train=True, transform=test_transform)
        # print("transform_query:", transform_query)
    else:
        queryset = datasets.__dict__[queryset_name](train=True, transform=test_transform)
    print("query_set:", len(queryset))

    # ----------- Initialize blackbox
    blackbox_dir = params['victim_model_dir']
    blackbox, num_classes = Blackbox.from_modeldir_split(blackbox_dir, device)  # 这里可以获得victim_model
    blackbox.eval()

    # ----------- Initialize adversary
    batch_size = params['batch_size']

    # test_valid_datasets = datasets.__dict__.keys()

    if test_dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    test_dataset = datasets.__dict__[test_dataset_name]
    testset = test_dataset(train=False, transform=test_transform)  # 这里是可以下载的

    model_parser = argparse.ArgumentParser(description='PyTorch ImageNet Testing')
    model_args = parser_params.add_parser_params(model_parser)

    # ----------- Set up queryset:总数据库
    print("\ndownload queryset dataset")  # 数据加载

    shadow_attack_model = None
    print('=> constructing transfer set...')
    adversary = GetAdversary(blackbox, queryset, batch_size=batch_size)  # 新建了一个类
    # ----------- get #'initial_seed'
    if params['sampling_method'] == 'label':
        transferset_o, pre_idxset_ = adversary.get_transferset(k=params['initial_seed'],
                                                               sampling_method=params['sampling_method'],
                                                               shadow_attack_model=shadow_attack_model)
    elif cfg.use_default_initial:
        initial_seed = osp.join(cfg.transfer_set_out_dir, cfg.queryset,
                                "initial_idxset_{}.npy".format(cfg.DEFAULT_SEED))
        # "initial_idxset_{}.npy".format(cfg.DEFAULT_SEED), "{}.npy".format(sm_m)
        pre_idxset_ = np.load(initial_seed)[:cfg.initial_seed]
        transferset_o, pre_idxset_ = adversary.get_transferset(k=params['initial_seed'],
                                                               sampling_method='use_default_initial',
                                                               shadow_attack_model=shadow_attack_model,
                                                               initial_seed=pre_idxset_)
    else:
        transferset_o, pre_idxset_ = adversary.get_transferset(k=params['initial_seed'],
                                                               sampling_method='initial_seed',
                                                               shadow_attack_model=shadow_attack_model)
        setting_path = osp.join(cfg.transfer_set_out_dir, cfg.queryset)
        if not os.path.exists(setting_path):
            os.makedirs(setting_path)
        np.save(setting_path + "//initial_idxset_" + str(cfg.DEFAULT_SEED), pre_idxset_)

    # change_to_trainable_set
    transferset = samples_to_transferset(transferset_o, budget=len(transferset_o), transform=transform_query)

    # sm_set = ['membership_attack,unsupervised', 'membership_attack,conf_shadow_model', 'membership_attack,shadow_model', 'random,']
    # sm_set = ['membership_attack,conf_shadow_model', 'membership_attack,shadow_model']
    #remark: ini is wrong, stop at ini.
    sm_set = ['cifar,,ini', 'cifar,,base'] #'mnist_tl,,over',
    for sm_m in sm_set:
        start_time = []
        start_time.append(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))#1
        exp = sm_m.split(',')
        params['sampling_method'] = exp[0]
        params['ma_method'] = exp[1]
        params['second_sss'] = exp[2]

        params['strategy'] = exp[2]
        # ----------- Set up attack model
        #attack_model_name = params['model_arch']
        if params['strategy'] == 'base':
            attack_model_name = 'lenet_tl'#'resnet18'
        else:
            attack_model_name = 'over_lenet_tl'#'over_resnet18'
        pretrained = params['pretrained']

        print("\n\n\n ----------------start {}".format(sm_m))

        print('=> Training at budget = {}'.format(len(transferset)))
        start_time.append(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))#this means the end of read initial seed #2

        # ----------- shadow_model
        criterion_train = model_utils.soft_cross_entropy
        if (cfg.imp_vic_mem and cfg.vic_mem_method =='shadow_model') or params['second_sss'] == 'shadow_model' or (
                params['sampling_method'] == 'membership_attack' and params['ma_method'] == 'shadow_model'):
            # # 取10000，10000
            transferset_shadow = adversary.transfer_to_shadow_set(transferset_o)[:2000]
            #need to add !!!!
            if cfg.augment_mi:
                print("====shadow model use augment mi")
                transferset_shadow = adversary.augment_shadow_set(transferset_shadow)

            shadow_set = samples_to_transferset(transferset_shadow, budget=len(transferset_shadow),
                                                transform=transform_query) #have transformation when training!

            shadow_set_2 = samples_to_transferset(transferset_shadow, budget=len(transferset_shadow),
                                                  transform=test_transform) #put it in again

            # shadow_queryset
            s_queryset_name = cfg.shadow_queryset
            s_queryset_names = s_queryset_name.split(',')
            print("valid_datasets: ", valid_datasets)
            for i, qname in enumerate(s_queryset_names):
                if qname.find("-") > -1:
                    qname = qname.split("-")[0]
                if qname not in valid_datasets:  # 几大data family
                    raise ValueError('Dataset not found. Valid arguments = {}, qname= {}'.format(valid_datasets, qname))

            if s_queryset_name == 'ImageFolder':
                assert params['root'] is not None, 'argument "--root ROOT" required for ImageFolder'
                s_queryset = datasets.__dict__[queryset_name](root=params['root'], transform=test_transform)
            elif len(s_queryset_names) > 1:  # 拥有多个dataset
                qns = "ChainMNIST"
                for qn in s_queryset_names:
                    if qn.find("CIFAR10") == 0:
                        qns = "ChainCIFAR"
                        break
                s_queryset = datasets.__dict__[qns](chain=s_queryset_names, train=True,
                                                    transform=test_transform)
                # print("transform_query:", transform_query)
            else:
                s_queryset = datasets.__dict__[queryset_name](train=True, transform=test_transform)
            print("query_set:", len(queryset))
            # transform_query = datasets.modelfamily_to_transforms[modelfamily]['train']
            s_adversary = GetAdversary(blackbox, s_queryset, batch_size=batch_size) #新建了一个类

            shadow_mode_path = osp.join(shadow_model_dir, sm_m)
            if not osp.exists(shadow_mode_path):
                os.makedirs(shadow_mode_path)
            shadow_log_path = osp.join(shadow_mode_path, '{}.log.tsv'.format(sm_m))

            s_checkpoint_suffix = '{}'.format(len(shadow_set))

            if cfg.read_shadow_from_path:
                shadow_model, _ = Blackbox.from_modeldir_split_attack_mode(shadow_mode_path, 'checkpoint_2000.pth.tar',device)

            else:
                if cfg.transfer_mi:
                    shadow_model = zoo.get_net(attack_model_name, modelfamily, num_classes=num_classes)
                    shadow_model = shadow_model.to(device)
                    optimizer1 = get_optimizer(shadow_model.parameters(), params['optimizer_choice'], **params)

                    transmi_path = osp.join(cfg.TRANSMI_DIR, 'checkpoint.pth.tar')
                    checkpoint_trans = torch.load(transmi_path)
                    # print("keys:", checkpoint_trans['state_dict'].keys())
                    # here we cant
                    for key in checkpoint_trans['state_dict'].keys():
                        if key.find('block3') == 0 or key.find("bn1") == 0 or key.find('fc') == 0:
                            print(key)
                            # covered by random
                            checkpoint_trans['state_dict'][key] = shadow_model.state_dict()[key]
                    shadow_model.load_state_dict(checkpoint_trans['state_dict'])
                    # checkpoint_suffix_s = '{}'.format(len(shadow_set))
                    print("start training shadow_model")
                    # *****************
                    model_utils.train_model(model=shadow_model, trainset=shadow_set, out_path=shadow_mode_path,
                                            blackbox=blackbox, testset=testset,
                                            criterion_train=criterion_train,
                                            checkpoint_suffix=s_checkpoint_suffix, device=device,
                                            optimizer=optimizer1,
                                            s_m=sm_m, args=model_args, **params)

                    p_shadow = argparse.ArgumentParser(description='Train a model')
                    args_shadow = p_shadow.parse_args()
                    p_save = vars(args_shadow)
                    p_save['model_arch'] = cfg.victim_model_arch
                    p_save['num_classes'] = num_classes
                    p_save['dataset'] = cfg.test_dataset
                    p_save['created_on'] = str(datetime.datetime.now())
                    p_save['start_point'] = str(cfg.start_shadow)
                    p_save['start_point_out'] = str(cfg.start_shadow_out)
                    p_save['query_dataset'] = str(cfg.queryset)
                    s_params_out_path = osp.join(shadow_mode_path, 'params.json')
                    with open(s_params_out_path, 'w') as jf:
                        json.dump(p_save, jf, indent=True)
                    print("start training shadow_attack_model")
                    shadow_model = Blackbox(shadow_model, device, 'probs')
                else:
                    num_classes = 10  # 先设一个，对mnist
                    p_shadow = argparse.ArgumentParser(description='Train a model')
                    args_shadow = p_shadow.parse_args()
                    p_save = vars(args_shadow)
                    p_save['model_arch'] = cfg.victim_model_arch
                    p_save['num_classes'] = num_classes
                    p_save['dataset'] = cfg.test_dataset

                    shadow_model = zoo.get_net(attack_model_name, modelfamily, num_classes=num_classes)
                    shadow_model = shadow_model.to(device)

                    optimizer2 = get_optimizer(shadow_model.parameters(), params['optimizer_choice'], **params)
                    # s_checkpoint_suffix = '{}'.format(len(shadow_set))
                    print("##############start training shadow_model#####################")
                    # ep_temp = params['epochs']
                    # params['epochs'] = 50
                    if not cfg.read_attack_mia_model_from_path:
                        model_utils.train_model(model=shadow_model, trainset=shadow_set, out_path=shadow_mode_path,
                                            blackbox=blackbox, testset=testset,
                                            criterion_train=criterion_train,
                                            checkpoint_suffix=s_checkpoint_suffix, device=device,
                                            optimizer=optimizer2,
                                            s_m=sm_m, args=model_args, **params)
                    # params['epochs'] = ep_temp
                    shadow_model = Blackbox(shadow_model, device, 'probs')

                    p_save['created_on'] = str(datetime.datetime.now())
                    p_save['start_point'] = str(cfg.start_shadow)
                    p_save['start_point_out'] = str(cfg.start_shadow_out)
                    p_save['query_dataset'] = str(cfg.queryset)
                    s_params_out_path = osp.join(shadow_mode_path, 'params.json')
                    with open(s_params_out_path, 'w') as jf:
                        json.dump(p_save, jf, indent=True)
                    print("start training shadow_attack_model")

            transferset_shadow_out = adversary.get_transferset_shadow_out(
                shadow_out_len=len(shadow_set))  # 不去管ma_method
            shadow_set_out = samples_to_transferset(transferset_shadow_out, budget=len(transferset_shadow_out),
                                                    transform=test_transform)
            print("shadow_model used out of training dataset:", len(shadow_set_out))
            if cfg.read_attack_mia_model_from_path:
                shadow_attack_model = load_bi_classifier(shadow_model=shadow_model, shadow_set=shadow_set_2,
                                                  shadow_set_out=shadow_set_out, device=device,
                                                  shadow_model_path=shadow_mode_path)
            else:
                shadow_attack_model = train_bi_classifier(shadow_model=shadow_model, shadow_set=shadow_set_2,
                                                      shadow_set_out=shadow_set_out, device=device, shadow_model_path=shadow_mode_path)  #we use not transfered data

            # Here test
            if cfg.mi_test:
                len_test = 20000
                y_mi_test, y_mi_target = s_adversary.get_transferset_shadow_test(k=len_test)
                confidence = []
                for input_batch, _ in classifier.iterate_minibatches(inputs=y_mi_test, targets=y_mi_target,
                                                                     batch_size=cfg.batch_size, shuffle=False):
                    # print("shadow_model_attack: input_batch:", input_batch.shape)
                    input = clipDataTopX(input_batch, top=3)
                    # top= [i[0]>0.5 for i in input]
                    # print("top", top)
                    pred = shadow_attack_model(input)  # output
                    confidence.append([p[1] for p in pred])
                confidence = np.concatenate(confidence)
                esti_training_50 = []
                esti_training_55 = []
                esti_training_60 = []
                esti_training_65 = []
                esti_no_training_50 = []
                esti_no_training_55 = []
                esti_no_training_60 = []
                esti_no_training_65 = []
                for idx, c in enumerate(confidence):
                    if c > 0.5:
                        esti_training_50.append(idx)
                    else:
                        esti_no_training_50.append(idx)
                    if c > 0.55:
                        esti_training_55.append(idx)
                    else:
                        esti_no_training_55.append(idx)
                    if c > 0.6:
                        esti_training_60.append(idx)
                    else:
                        esti_no_training_60.append(idx)
                    if c > 0.65:
                        esti_training_65.append(idx)
                    else:
                        esti_no_training_65.append(idx)
                print("threshold=0.5")
                tp1 = np.sum(np.asarray(esti_training_50) < len_test / 2)
                acc1 = np.sum(np.asarray(esti_training_50) < len_test / 2) + np.sum(
                    np.asarray(esti_no_training_50) >= len_test / 2)
                print("true_positive:", tp1, '/', len(esti_training_50))
                print("acc:", acc1 / len_test)

                print("threshold=0.55")
                tp2 = np.sum(np.asarray(esti_training_55) < len_test / 2)
                acc2 = np.sum(np.asarray(esti_training_55) < len_test / 2) + np.sum(
                    np.asarray(esti_no_training_55) >= len_test / 2)
                print("true_positive:", tp2, '/', len(esti_training_55))
                print("acc:", acc2 / len_test)

                print("threshold=0.6")
                tp3 = np.sum(np.asarray(esti_training_60) < len_test / 2)
                acc3 = np.sum(np.asarray(esti_training_60) < len_test / 2) + np.sum(
                    np.asarray(esti_no_training_60) >= len_test / 2)
                print("true_positive:", tp3, '/', len(esti_training_60))
                print("acc:", acc3 / len_test)

                print("threshold=0.65")
                tp4 = np.sum(np.asarray(esti_training_65) < len_test / 2)
                acc4 = np.sum(np.asarray(esti_training_65) < len_test / 2) + np.sum(
                    np.asarray(esti_no_training_65) >= len_test / 2)
                print("true_positive:", tp4, '/', len(esti_training_65))
                print("acc:", acc4 / len_test)

                with open(shadow_log_path, 'a') as af:
                    columns = ['shadow_model_acc_len', 'acc0.5','tp','esti_all', 'acc0.55', 'tp','esti_all', 'acc0.6','tp','esti_all', 'acc0.65', 'tp','esti_all']
                    af.write('\t'.join(columns) + '\n')
                    train_cols = [len(shadow_set), acc1 / len_test, tp1, esti_training_50,
                                  acc2 / len_test, tp2, esti_training_55, acc3 / len_test, tp3, esti_training_60,
                                  acc4 / len_test, tp4, esti_training_60]
                    af.write('\t'.join([str(c) for c in train_cols]) + '\n')

        #vmi:
        if cfg.imp_vic_mem:
            transferset_o = adversary.mia_on_query_result(mi_attacker=shadow_attack_model)
            transferset = samples_to_transferset(transferset_o, budget=len(transferset_o), transform=transform_query)

        start_time.append(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) #3

        # -----initial
        # torch.manual_seed(cfg.DEFAULT_SEED)
        torch.cuda.manual_seed(cfg.DEFAULT_SEED)  # 使 model的初始化方式一样
        b = 0
        out_path = osp.join(attack_model_dir, cfg.queryset)
        if not osp.exists(out_path):
            os.makedirs(out_path)
        time_out_path = osp.join(out_path, 'time-{}.json'.format(sm_m))
        json_head = ['start_project', 'end_of_initial_seed', 'end_of_initial_shadow_model', 'start_it_train',
                     'end_it_train',
                     'end_select_next_seed', 'end_shadow_model']
        with open(time_out_path, 'a') as jf: #overwriten
            json.dump(json_head, jf, indent=True)
            json.dump(start_time, jf, indent=True)

        criterion_train = model_utils.soft_cross_entropy
        if params['iterative']:  # start iteration train
            for it in range(params['num_iter']):
                it_time = []
                print('\n---------------start {} iteration'.format(it))
                # ----- process data
                if it == 0:
                    b = b + params['initial_seed']
                else:
                    b = b + params['k']

                # ----- restart attack_model
                attack_model = zoo.get_net(attack_model_name, modelfamily, pretrained, num_classes=num_classes)
                # attack_model= Blackbox.from_modeldir_resume(out_path, 'checkpoint_{}.pth.tar'.format(b), device) #resume
                attack_model = attack_model.to(device)
                if params['strategy'] == 'ini':
                    # N_ini_query = 80000
                    # w1_dir1 = torch.load(f'./data_ini/svhn/w1_1-0-{N_ini_query}.pt').to(device)
                    # w1_dir2 = torch.load(f'./data_ini/svhn/w1_1-1-{N_ini_query}.pt').to(device)
                    # w1_dir3 = torch.load(f'./data_ini/svhn/w1_1-2-{N_ini_query}.pt').to(device)
                    # w1_dir4 = torch.load(f'./data_ini/svhn/w1_1-3-{N_ini_query}.pt').to(device)
                    # w1_dir5 = torch.load(f'./data_ini/svhn/w1_1-4-{N_ini_query}.pt').to(device)

                    # w1_dir1 = torch.load(f'./data_ini/cifar10/w1_cifar-0-{N_ini_query}.pt')
                    # w1_dir2 = torch.load(f'./data_ini/cifar10/w1_cifar-1-{N_ini_query}.pt')
                    # w1_dir3 = torch.load(f'./data_ini/cifar10/w1_cifar-2-{N_ini_query}.pt')
                    # w1_dir4 = torch.load(f'./data_ini/cifar10/w1_cifar-3-{N_ini_query}.pt')
                    # w1_dir5 = torch.load(f'./data_ini/cifar10/w1_cifar-4-{N_ini_query}.pt')
                    # nc=3
                    # init_w1 = over_initial_cnn(w1_dir1, w1_dir2, w1_dir3, w1_dir4, w1_dir5, channel=nc)
                    # attack_model.conv1.weight = Parameter(init_w1)
                    N_ini_query = 2000
                    w1_dir1 = torch.load(f'./data_ini/mnist_tl/w_mnist_tl0-0-{N_ini_query}.pt')
                    w1_dir2 = torch.load(f'./data_ini/mnist_tl/w_mnist_tl0-1-{N_ini_query}.pt')
                    w1_dir3 = torch.load(f'./data_ini/mnist_tl/w_mnist_tl0-2-{N_ini_query}.pt')
                    w1_dir4 = torch.load(f'./data_ini/mnist_tl/w_mnist_tl0-3-{N_ini_query}.pt')
                    w1_dir5 = torch.load(f'./data_ini/mnist_tl/w_mnist_tl0-4-{N_ini_query}.pt')
                    init_w1 = over_initial(w1_dir1, w1_dir2, w1_dir3, w1_dir4, w1_dir5).to(device)
                    attack_model.fc1.weight = Parameter(init_w1)

                optimizer = get_optimizer(attack_model.parameters(), params['optimizer_choice'], **params)
                # print(params)

                checkpoint_suffix = '{}'.format(b)
                # 训练标准, function, will return a cross-entropy loss

                # 训练 攻击模型  # attack_model_dir 是 attack_model_output_dir # testset comes from mnist-test # checkpoint_suffix 后缀
                # ----- Train #transferset_是用于训练的数据
                it_time.append(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) #4

                if False:#it==0:
                    # attack_model = Blackbox.from_modeldir_split_attack_mode(out_path,
                    #                                                            'checkpoint_{}.pth.tar'.format(
                    #                                                                cfg.initial_seed), device)
                    print("do nothing")
                else:
                    model_utils.train_model(model=attack_model, trainset=transferset, out_path=out_path,
                                            blackbox=blackbox, testset=testset,
                                            criterion_train=criterion_train,
                                            checkpoint_suffix=checkpoint_suffix, device=device,
                                            optimizer=optimizer,
                                            s_m=sm_m, args=model_args, imp_vic_mem=cfg.imp_vic_mem, mode=params['strategy'], **params)

                it_time.append(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) #5
                # ----- Choose next bunch of queried data
                if (it + 1) < params['num_iter']:
                    attack_model, _ = Blackbox.from_modeldir_split_attack_mode(out_path,
                                                                               'checkpoint_{}.pth.tar'.format(
                                                                                   checkpoint_suffix), device)
                    adversary.set_attack_model(attack_model, device)  # 将attack model输入

                    if params['sampling_method'] == 'membership_attack':  # membership
                        print("params['ma-method']:", params['ma_method'])
                        if params['ma_method'] == 'shadow_model':
                            transferset_o, pre_idxset_ = adversary.get_transferset(k=params['k'],
                                                                                   sampling_method=params[
                                                                                       'sampling_method'],
                                                                                   ma_method=params['ma_method'],
                                                                                   pre_idxset=pre_idxset_,
                                                                                   shadow_attack_model=shadow_attack_model,
                                                                                   second_sss=params['second_sss'],
                                                                                   it=it)
                        elif params['ma_method'] == 'unsupervised':
                            transferset_o, pre_idxset_ = adversary.get_transferset(k=params['k'],
                                                                                   sampling_method=params[
                                                                                       'sampling_method'],
                                                                                   ma_method=params['ma_method'],
                                                                                   pre_idxset=pre_idxset_)
                        else:
                            transferset_o, pre_idxset_ = adversary.get_transferset(k=params['k'],
                                                                                   sampling_method=params[
                                                                                       'sampling_method'],
                                                                                   ma_method=params['ma_method'],
                                                                                   pre_idxset=pre_idxset_)
                    else:  # kcenter; adaptive; adversarial
                        transferset_o, pre_idxset_ = adversary.get_transferset(k=params['k'], sampling_method=params[
                            'sampling_method'],
                                                                               pre_idxset=pre_idxset_,
                                                                               shadow_attack_model=shadow_attack_model,
                                                                               second_sss=params[
                                                                                   'second_sss'])  # 不去管ma_method

                    print("choose_finished: transformset_o:", len(transferset_o))
                    # change_to_trainable_set
                    transferset = samples_to_transferset(transferset_o, budget=len(transferset_o),
                                                         transform=transform_query)
                    print('=> Training at budget = {}'.format(len(transferset)))
                    # update(shadow_model)
                    it_time.append(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) #6

                    setting_path = osp.join(cfg.transfer_set_out_dir, cfg.queryset)
                    if not os.path.exists(setting_path):
                        os.makedirs(setting_path)
                    np.save(setting_path + "//" + sm_m, pre_idxset_)

                    it_time.append(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))  #7

                    # vmi:
                    if cfg.imp_vic_mem:
                        print("we do have imp_vic_mem")
                        transferset_o = adversary.mia_on_query_result(mi_attacker=shadow_attack_model)
                        transferset = samples_to_transferset(transferset_o, budget=len(transferset_o),
                                                             transform=transform_query)

                with open(time_out_path, 'a') as jf:  # change
                    json.dump(it_time, jf, indent=True)

            print("pre+idexset", pre_idxset_)  # 显示data的idx
        else:
            print("No implemented")


        # ----- Store transferomet_o to transferset.pickle
        # out_path = params['out_dir']  # 输出adv的结果
        # transfer_out_path = osp.join(out_path, 'transferset.pickle')  # 序列化储存格式
        # with open(transfer_out_path, 'wb') as wf:
        #     pickle.dump(transferset, wf)
        # print('=> transfer set ({} samples) written to: {}'.format(len(transferset), transfer_out_path))

        # ----- Store arguments to params.json
        # params['created_on'] = str(datetime.datetime.now())
        # params_out_path = osp.join(attack_model_dir, 'params.json')

if __name__ == '__main__':
    main()

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
from tensor_train_for_maze_dfme import TransferSetGaussian
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

    def mia_on_query_result(self, mi_attacker=None):
        # use self.pre_idxset
        # use self.queryset
        self.transferset = []
        k = 0 # k is vain
        for b in range(int(np.ceil(len(self.pre_idxset) / self.batch_size))):  # s
            # x_b = x_t[b * self.batch_size: min((1 + b) * self.batch_size, len(s))].to(self.blackbox.device)
            c_idx = self.pre_idxset[
                    b * self.batch_size: min((1 + b) * self.batch_size, len(self.pre_idxset))]  # chosed_idx
            #all choosed idx set
            x_b = torch.stack([self.queryset[int(i)][0] for i in c_idx]).to(self.blackbox.device)
            y_b = self.blackbox(x_b).cpu()
            # print("y_b,", y_b)
            if cfg.imp_vic_mem:
                if cfg.vic_mem_method == 'unsupervised':
                    print("use imp vic mem -- unsupervised")
                    unsuperY = []
                    unsuper_idx_set = set(range(35000, 35000 + int(cfg.unsuper_data)))  # 184800
                    for u in range(int(np.ceil(len(unsuper_idx_set) / self.batch_size))):
                        # download_x
                        # idx来自idx_set
                        uidx = list(unsuper_idx_set)[
                               u * self.batch_size: min((1 + u) * self.batch_size, len(unsuper_idx_set))]
                        # print("uidx:", uidx)
                        x_u = torch.stack([self.queryset[iu][0] for iu in uidx]).to(self.attack_device)
                        y_u = self.attack_model(x_u).cpu()  # not training data.
                        # X_rest.append(x_u.cpu())
                        # yu_top_1 = clipDataTopX(y_u, top=1)
                        # unsuperY.append(yu_top_1)
                        unsuperY.append(y_u.numpy())
                        # just get unsupery
                    self.unsuperY = np.concatenate(unsuperY)
                    vsss = UnsupervisedMemberAttack(k, y_b.numpy(), tolerant_rate=0.06, unsuperY=self.unsuperY)
                    #here just use a unsupervsed way- it need a pre-trained attack model
                else:
                    shadow_attack_model = mi_attacker
                    vsss = ShadowModelMemberAttack(k, shadow_attack_model, y_b)  # who
                vs = vsss.get_subset()
                # member_idx = [c_idx[int(ivs)] for ivs in vs]#vs is index
                print("the part of 'true' membership is:", len(vs) / len(c_idx))
                y_member = torch.ones(y_b.shape)
                y_member[vs] = 5 #if member is 1
                y_bf = torch.stack((y_b, y_member), dim=1)
                # c_idx = [c_idx]
                # y_b = np.asarray(y_b)
                # y_member = [y_b[int(mi2)] for mi2 in vs]#y_b is the query result
                # c_idx.append(member_idx)
                # if len(y_member) > 0:
                    # y_b = torch.tensor(np.vstack((y_b, y_member)))
                # else:
                    # y_b = torch.tensor(y_b)
                # c_idx = np.concatenate(c_idx).astype(int)

            if hasattr(self.queryset, 'samples'):
                # Any DatasetFolder (or subclass) has this attribute # Saving image paths are space-efficient
                img_p = [self.queryset.samples[i][0] for i in c_idx]  # Image paths
            else:
                # Otherwise, store the image itself # But, we need to store the non-transformed version
                # img_t = [self.queryset.data[i] for i in c_idx]
                # img_t = self.queryset.getitemsinchain(c_idx)
                if len(self.queryset.data) <= 3:
                    # print("\nwe use a mnist chain")
                    img_p, _ = self.queryset.getitemsinchain(c_idx)
                else:
                    img_p = [self.queryset.data[i] for i in c_idx]
                # if isinstance(self.queryset.data[0], torch.Tensor):
                #     img_p = [x.numpy() for x in img_p]
                # if isinstance(img_p[0], torch.Tensor):
                #     img_p = [x.numpy() for x in img_p]

            for m in range(len(c_idx)):
                img_p_i = img_p[m].numpy() if isinstance(img_p[m], torch.Tensor) else img_p[m]
                img_t_i = img_p_i.squeeze() if isinstance(img_p_i, np.ndarray) else img_p_i
                self.transferset.append((img_t_i, y_bf[m].squeeze()))  # self.transferset
                # self.transfery.append(y_b[m].squeeze().numpy())

        return self.transferset


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
    valid_datasets = datasets.__dict__.keys()

    """
    Set up testset
    """
    test_dataset_name = params['testdataset']  # 用的是MNIST test
    test_modelfamily = datasets.dataset_to_modelfamily[test_dataset_name]
    test_transform = datasets.modelfamily_to_transforms[test_modelfamily]['test']
    print("test_transform:", test_transform.__dict__.keys())

    """
    Initialize
    """
    blackbox_dir = params['victim_model_dir']
    blackbox, num_classes = Blackbox.from_modeldir_split(blackbox_dir, device)  # 这里可以获得victim_model
    blackbox.eval()

    if test_dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    test_dataset = datasets.__dict__[test_dataset_name]
    testset = test_dataset(train=False, transform=test_transform)  # 这里是可以下载的

    model_parser = argparse.ArgumentParser(description='PyTorch ImageNet Testing')
    model_args = parser_params.add_parser_params(model_parser)

    sm_set = ['cifar10,maze,ini,layer-wise']
    for sm_m in sm_set:
        start_time = []
        start_time.append(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))#1
        exp = sm_m.split(',')
        params['sampling_method'] = exp[0]
        params['me_method'] = exp[1]
        params['strategy'] = exp[2]

        print("\n\n\n ----------------start {}".format(sm_m))
        """
        reload dfme/maze generated data
        """
        query_num = [20480000, 40960000, 50176000]
        # query_num = [10240000, 15360000, 20480000, 30720000, 40960000, 46080000, 50073600, 50176000]
        print("\n\n\n ----------------start {}".format(sm_m))
        if params['me_method'] == 'maze' and params['strategy'] == 'ini':
            x_train = torch.load(f"./data_dfme/x_batch_t-cifar10_maze_ini_{query_num[0]}.pt").cpu()
            for q_i in range(1, len(query_num)):
                x_train_2 = torch.load(f"./data_dfme/x_batch_t-cifar10_maze_ini_{query_num[q_i]}.pt").cpu()
                x_train = torch.cat((x_train, x_train_2))
        elif params['me_method'] == 'maze' and params['strategy'] == 'over':
            x_train = torch.load("./data_dfme/x_batch_t-cifar10_maze_over_10560000.pt").cpu()
            for q_i in range(1, query_num):
                x_train_2 = torch.load(f"./data_dfme/x_batch_t-cifar10_maze_over_{query_num[q_i]}.pt").cpu()
                x_train = torch.cat((x_train, x_train_2))

        query_batch=10
        with torch.no_grad():
            for i_q in range(int(len(x_train) / query_batch)):
                x_train_temp = x_train[i_q * query_batch: (i_q + 1) * query_batch, :, :, :]
                y_train_temp = blackbox(x_train_temp.to(device)).cpu()
                if i_q == 0:
                    y_train = y_train_temp
                else:
                    y_train = torch.cat((y_train, y_train_temp), dim=0)

        print('=> x_train budget = {}'.format(y_train.shape))
        transferset = TransferSetGaussian(x_train.cpu(), y_train.cpu())
        print('=> Training at budget = {}'.format(len(transferset)))
        start_time.append(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))#this means the end of read initial seed #2

        # -----initial
        torch.cuda.manual_seed(cfg.DEFAULT_SEED)  # 使 model的初始化方式一样
        b = 0
        out_path = osp.join(attack_model_dir, cfg.test_dataset)# cfg.queryset
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

        """
        load dfme/maze generated model
        """
        if params['me_method'] == 'maze' and params['strategy'] == 'ini':
            attack_model, _ = Blackbox.from_modeldir_split_attack_mode(out_path,
                                                                       'checkpoint_cifar10_maze_ini_50176000.pth.tar',
                                                                       device)  # checkpoint_10000_ini_0

        elif params['me_method'] == 'dfme' and params['strategy'] == 'ini':
            attack_model, _ = Blackbox.from_modeldir_split_attack_mode(out_path,
                                                                       'checkpoint_cifar10_maze_over_49766400.pth.tar',
                                                                       device)  # resume

        attack_model = attack_model.get_model()
        attack_model = attack_model.to(device)

        optimizer = get_optimizer(attack_model.parameters(), params['optimizer_choice'], **params)
        checkpoint_suffix = '{}'.format(b)

        """
        layer-wise-training strategy
        """
        model_utils.rec_train_model(model=attack_model, trainset=transferset, out_path=out_path,
                                    blackbox=blackbox, testset=testset,
                                    criterion_train=criterion_train,
                                    checkpoint_suffix=checkpoint_suffix, device=device,
                                    optimizer=optimizer,
                                    s_m=sm_m, args=model_args, imp_vic_mem=cfg.imp_vic_mem, mode=params['strategy'],
                                    **params)


if __name__ == '__main__':
    main()

import argparse
import copy
import json
import math
import os
from datetime import datetime

import torch
from sklearn.linear_model import ridge_regression
from torch import autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import datasets
import parser_params
import utils as me_utils
import zoo
from blackbox import Blackbox
import config as cfg
from converge_main import get_optimizer, samples_to_transferset
from gmm import GaussianMixture
import numpy as np

from model_scheduler import soft_cross_entropy
from no_tenfact import no_tenfact
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import os.path as osp
import mebooster.utils.model_scheduler_special as model_utils
from nontarget_random_fgsm import NonTargetFGSM


class TransferSetGaussian(Dataset):
    def __init__(self, x_train, y):
        self.data = x_train
        self.targets = y

    def __getitem__(self, index):
        x, target = self.data[index, :], self.targets[index]

        return x, target

    def __len__(self):
        return len(self.data)

def get_teacher_model(device):
    blackbox_dir = cfg.VICTIM_DIR
    blackbox, num_classes = Blackbox.from_modeldir_split(blackbox_dir, device)
    blackbox.eval()
    return blackbox


def over_initial(student_model, w1_dir, d, k, over_factor=3, device='cuda:0'):
    #over_factor
    w1 = w1_dir
    # mag1 = torch.abs(torch.randn([k])).to(device)
    # mag2 = -torch.abs(torch.randn([k])).to(device)
    # mag3 = torch.abs(torch.randn([k])).to(device)
    # mag4 = -torch.abs(torch.randn([k])).to(device)
    # mag5 = torch.abs(torch.randn([k])).to(device)

    # mag1 = torch.randn([k]).to(device)
    # mag2 = torch.randn([k]).to(device)
    # mag3 = torch.randn([k]).to(device)
    # mag4 = torch.randn([k]).to(device)
    # mag5 = torch.randn([k]).to(device)

    init_w1_1 = (w1/torch.norm(w1, dim=1).view(-1, 1)) * math.sqrt(2/d)#mag1# * w1 #d*k torch.randn([k, d]).to(device)
    init_w1_2 = (w1/torch.norm(w1, dim=1).view(-1, 1)) * math.sqrt(2/d)#mag2#torch.randn([k, d]).to(device) * math.sqrt(2/d)#mag2 * w1
    init_w1_3 = (w1/torch.norm(w1, dim=1).view(-1, 1)) * math.sqrt(2/d)#mag3#torch.randn([k, d]).to(device) * math.sqrt(2/d)

    init_w1_4 = (w1 / torch.norm(w1, dim=1).view(-1, 1)) * math.sqrt(2/d)#mag4
    init_w1_5 = (w1 / torch.norm(w1, dim=1).view(-1, 1)) * math.sqrt(2/d)#mag5

    # print("init_w1_1,", init_w1_1-init_w1_2)
    # print("init_w1_2,", init_w1_2.shape)
    # init_w1_3 = torch.randn_like(w1_dir).to(device)#mag3 * w1

    init_w1 = torch.vstack((init_w1_1.T, init_w1_2.T, init_w1_3.T, init_w1_4.T, init_w1_5.T)) #init_w1_4.T, init_w1_5.T
    return init_w1


def test_utils(model, test_loader):
    # test_err = 0.0
    # total = len(test_loader.dataset)
    # for inputs, targets in test_loader:
    #     inputs, targets = inputs.to(device), targets.to(device)
    #     test_err += torch.sum(torch.abs(model(inputs) - targets), dim=0)
    # test_err = test_err/total
    # print("test_err", test_err.cpu().detach().numpy())
    # test
    model.eval()
    test_acc_item = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        _, predicted = model(inputs).max(1)
        _, gt_label = targets.max(1)
        test_acc_item += predicted.eq(gt_label).sum().item()

        total += targets.size(0)
    acc = test_acc_item / total
    print("test_acc:", acc * 100.)
    return acc

def train_model(model, train_loader, test_loader, optimizer, criterion_train, epochs, model_out_path):
    best_acc = 0.
    epoch_size = len(train_loader.dataset)
    log_interval = 300
    for epoch in range(0, epochs):
        train_loss = 0.
        err = 0
        total = 0
        acc_item = 0.
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad = True

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion_train(outputs, targets)
            # loss = torch.sum((1 - targets) * torch.log(torch.abs(outputs) + 1e-16) + targets * torch.log(
            #     torch.abs(1 - outputs) + 1e-16))
            # print("loss", loss.shape)

            loss.backward(retain_graph=True)
            optimizer.step()

            train_loss += loss.item()
            # _, predicted = outputs.max(1)
            total += targets.size(0)
            # err += torch.sum(torch.abs(model(inputs) - targets), dim=0)
            prog = total / epoch_size
            exact_epoch = epoch + prog

            train_loss_batch = train_loss / total

            _, predicted = model(inputs).max(1)
            _, gt_label = targets.max(1)
            acc_item += predicted.eq(gt_label).sum().item()
            acc = acc_item / total

            if (batch_idx + 1) % log_interval == 0:
                print('[Train] Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.4f} ({}/{})'.format(
                    exact_epoch, batch_idx * len(inputs), len(train_loader.dataset),
                                 100. * batch_idx / len(train_loader),
                    loss.item(), acc*100, acc_item, total))

        test_acc=test_utils(model, test_loader)

        if test_acc >= best_acc:
            # print("small_err,", small_err)
            best_acc = acc
            state = {
                'epoch': 100,
                'arch': model.__class__,
                'state_dict': model.state_dict(),
                'best_acc': test_acc,
                'optimizer': optimizer,
                'created_on': str(datetime.now()),
                }
            torch.save(state, model_out_path)

def one_hot_tensor(label, class_num=10):
    one_hot = torch.zeros([1, class_num])
    # one_hot[:, label[0]] += .5
    # one_hot[:, label[1]] += .5
    one_hot[:, label] += 1.
    return one_hot

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
        self.fd_gradient = [] #forward difference
        self.no_training_in_initial = 0

    def _restart(self):
        np.random.seed(cfg.DEFAULT_SEED)
        torch.manual_seed(cfg.DEFAULT_SEED)
        torch.cuda.manual_seed(cfg.DEFAULT_SEED)

        self.idx_set = set(range(len(self.queryset))) #idx根据queryset进行更改的
        self.transferset = []

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

    # def get_synthesizing_from_fgsm(self, x_t, y_t, epsilon, it=0):
    #     method = NonTargetFGSM(self.attack_model.get_model())
    #     generate_result=[]
    #
    #     synthesizing_images, u=method.get_synthesizing_set(x_t, y_t, epsilon)
    #     generate_result.append(synthesizing_images.detach().cpu().numpy())
    #     generate_result = np.concatenate(np.asarray(generate_result), axis=0)
    #     #query and add to transferset
    #     #and save
    #     generate_tensor = torch.tensor(generate_result).cuda()
    #     logger = Logger(model_name='FGSM', data_name='MNSIT')
    #     for saver in range(int(len(generate_tensor)/30)):
    #         save_tensor = generate_tensor[(saver*100):min(len(generate_tensor), (saver+1)*100)].cpu()
    #         outdir = './data/{}_{}/'.format(it, saver)
    #         logger.log_generate_images(save_tensor, len(save_tensor), 0, 0, 0,
    #                                outdir=outdir)#generate_number_k
    #     return generate_tensor, u

    def random_target(self, len=32, class_num=10):
        label = np.random.randint(class_num, size=[len, 2])
        targets = torch.stack([one_hot_tensor(label[i]) for i in range(len)]).view([-1, 10])
        targets = Variable(targets)
        return targets

    def get_transferset(self, k, sampling_method='random', ma_method='unsupervised', pre_idxset=[],
                        shadow_attack_model=None, device='cuda', second_sss=None, it=None, initial_seed=[]):
        self.sampling_method=sampling_method
        self.ma_method=ma_method

        dt=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("start_query_strategy:", dt)
        with tqdm(total=k) as pbar:
            if self.sampling_method == 'initial_seed' or self.sampling_method == 'training_data' or\
                self.sampling_method == 'label':
                for bs, _ in enumerate(range(0, k, self.batch_size)):  # 1,200;步长：8
                    #only choose the first 2000
                    self.q_idx_set = copy.copy(self.idx_set)
                    if len(initial_seed)>0:
                        idxs = initial_seed[bs*self.batch_size: min((bs+1)*self.batch_size, len(initial_seed))].astype(int)
                    else:
                        idxs = np.random.choice(list(self.q_idx_set), replace=False,
                                            size=min(self.batch_size, k-len(self.pre_idxset)))  # 8，200-目前拥有的transferset的大小。

                    self.idx_set = self.idx_set - set(idxs)
                    self.pre_idxset = np.append(self.pre_idxset, idxs)
                    # print("idx,", idxs)
                    if len(self.idx_set) == 0:
                        print('=> Query set exhausted. Now repeating input examples.')
                        self.idx_set = set(range(len(self.queryset)))

                    x_t = torch.stack([self.queryset[i][0] for i in idxs]).to(device)
                    # x_t = Variable(x_t, requires_grad=True)
                    y_t = self.blackbox(x_t)
                    # y_t = F.softmax(y_t, dim=1)

                    if hasattr(self.queryset, 'samples'):
                        img_t = [self.queryset.samples[i][0] for i in idxs]  # Image paths
                    else:
                        if len(self.queryset.data) <= 3:
                            img_t, gt_label = self.queryset.getitemsinchain(idxs)
                        else:
                            img_t = [self.queryset.data[i] for i in idxs]
                    y_t = y_t.detach().cpu()

                    for i in range(len(idxs)):
                        img_t_i = img_t[i].numpy() if isinstance(img_t[i], torch.Tensor) else img_t[i]
                        img_t_i = img_t_i.squeeze() if isinstance(img_t_i, np.ndarray) else img_t_i
                        self.transferset.append((img_t_i, y_t[i].squeeze()))

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
                        img_p = [self.queryset.samples[i][0] for i in c_idx]  # Image paths
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
        dt1 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("start_query_strategy:", dt1)

        return self.transferset, self.pre_idxset

if __name__ == '__main__':
    #step 2
    #init
    device = torch.device('cuda')
    # eps = 0.01
    # lam = 1

    # N_eval = 1000
    # N_train = 20000
    # N_query = 20000  # used to calcuate the score functions
    d = 10
    m1 = 8
    num_classes = 10
    over_factor = 5
    out_path = osp.join(cfg.attack_model_dir, 'tensor_train')
    epochs=50
    batch_size=32

    N_query = 20000
    dataset_name = 'MNIST'
    valid_datasets = datasets.__dict__.keys()
    print("valid_datasets:", valid_datasets)
    if dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    dataset = datasets.__dict__[dataset_name]

    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    print("modelfamily", modelfamily)

    train_transform = datasets.modelfamily_to_transforms[modelfamily]['train']
    test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    # trainset0 = dataset(train=True, transform=train_transform)  # , download=True
    # trainset = Subset(trainset0, np.asarray(range(N_query)))
    test_dataset = dataset(train=False, transform=test_transform)  # , download=True

    #teacher model
    teacher_model = get_teacher_model(device)  # fc1(weight, bias), fc2(weight, bias)
    teacher_model.eval()
    print("teacher_model", teacher_model)

    queryset = datasets.__dict__[dataset_name](train=True, transform=test_transform) #queryset_name
    #tranferset
    print('=> constructing transfer set...')
    adversary = GetAdversary(teacher_model, queryset, batch_size=batch_size)
    pre_idxset_ = []
    transferset_o, pre_idxset_ = adversary.get_transferset(k=N_query,
                                                           sampling_method='initial_seed',
                                                           initial_seed=pre_idxset_)
    setting_path = osp.join(cfg.transfer_set_out_dir, dataset_name, 'tensor_train')
    if not os.path.exists(setting_path):
        os.makedirs(setting_path)
    np.save(setting_path + "//initial_idxset_" + str(cfg.DEFAULT_SEED), pre_idxset_)
    trainset = samples_to_transferset(transferset_o, budget=len(transferset_o), transform=train_transform)
    print('=> Training at budget = {}'.format(len(trainset)))

    #student model
    student_model = zoo.get_net('over_snet', 'mnist', None, over_factor=over_factor, num_classes=num_classes)
    student_model = student_model.to(device)
    random_model = zoo.get_net('over_snet', 'mnist', None, over_factor=over_factor, num_classes=num_classes)
    random_model = random_model.to(device)
    base_model = zoo.get_net('snet', 'mnist', None, num_classes=num_classes)
    base_model = base_model.to(device)

    #load ini
    # scorepath = 'E:\\Yaxin\\Work2\\training_algorithm\\mebooster\\score_function'
    # X_train = torch.load(scorepath +'\\model\\x2.pt').to(device)
    # x_train = torch.load(cfg.VICTIM_DIR + '\\x_train.pt').to(device)

    # x_train = torch.randn(N_query, d)
    # y_o = teacher_model(x_train.to(device))  # not real train, 10000
    # y_train = torch.softmax(y_o, dim=1)

    w1_dir = torch.load('w1_dir.pt').to(device)
    init_w1 = over_initial(student_model, w1_dir, d, m1, over_factor, device) #torch
    student_model.fc1.weight = Parameter(init_w1)
    # basic_out_path = osp.join(out_path, 'basis')
    # base_model, _ = Blackbox.from_modeldir_split(basic_out_path, device)
    # base_model.eval()
    # student_model.conv1 = base_model.conv1
    # student_model.conv2 = base_model.conv2

    # print("student.w1", student_model.fc1.weight.shape)

    # x_eval = torch.randn(N_eval, d)#.to(device)
    # y_eval = teacher_model(x_eval.to(device)).cpu()
    #train
    student_model.train()
    random_model.train()
    base_model.train()
    # dataset = TransferSetGaussian(x_train.cpu(), y_train.cpu())
    # train_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True)

    # test_dataset = TransferSetGaussian(x_eval, y_eval)
    # test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, pin_memory=True)
    # criterion_train = soft_cross_entropy

    if not osp.exists(out_path):
        me_utils.create_dir(out_path)
    model_parser = argparse.ArgumentParser(description='PyTorch ImageNet Testing')
    model_args = parser_params.add_parser_params(model_parser)

    #Accuracy98.7
    # print("ini_model,", student_model)
    # model_out_path = osp.join(out_path,'init_over')
    # optimizer = get_optimizer(student_model.parameters(), 'sgd', lr=0.01, momentum=0.5)
    # # train_model(student_model, train_loader, test_loader, optimizer, criterion_train, epochs, model_out_path)
    # model_utils.train_model(model=student_model, ori_model=None, layer=None, out_path=model_out_path, blackbox=teacher_model,
    #                         trainset=trainset, testset=test_dataset, device=device, args=model_args, epochs=epochs)#, work_mode='victim_train'

   # print("random_model,", random_model)
  #  sec_model_out_path = osp.join(out_path,'rand_over')
   # sec_optimizer = get_optimizer(random_model.parameters(), 'sgd', lr=0.01, momentum=0.5)
   # # train_model(random_model, train_loader, test_loader, sec_optimizer, criterion_train, epochs, sec_model_out_path)
   # model_utils.train_model(model=random_model, ori_model=None, layer=None, out_path=sec_model_out_path, blackbox=teacher_model,
    #                        trainset=trainset, testset=test_dataset, device=device, args=model_args, epochs=epochs)#, work_mode='victim_train'

    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--dataset', metavar='DS_NAME', type=str, help='Dataset name', default='MNIST')
    parser.add_argument('--model_arch', metavar='MODEL_ARCH', type=str, help='Model name', default='over_snet') #gaussian_nn snet
    args = parser.parse_args()
    params = vars(args)
    #random
    basic_out_path = osp.join(out_path, 'basis')
    basic_optimizer = get_optimizer(base_model.parameters(), 'sgd', lr=0.01, momentum=0.5)
    # train_model(base_model, train_loader, test_loader, basic_optimizer, criterion_train, epochs, basic_out_path)
    model_utils.train_model(model=base_model, ori_model=None, layer=None, out_path=basic_out_path, blackbox=teacher_model,
                           trainset=trainset, testset=test_dataset, device=device, args=model_args, epochs=epochs)
    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(out_path, 'params.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)

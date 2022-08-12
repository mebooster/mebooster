from __future__ import print_function
import argparse, ipdb, json
import copy
import math
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
#import network
#from dataloader import get_dataloader
import os, random
import numpy as np
import torchvision
from pprint import pprint
from time import time

from torch.nn import Parameter
from torch.utils.data import DataLoader

import datasets
import zoo
from blackbox import Blackbox
from dfme_utils.approximate_gradients import *

import torchvision.models as models
from dfme_utils.my_utils import *
import config as cfg
from model_scheduler import soft_cross_entropy
from models import get_maze_model
import os.path as osp
import csv

from utils.utils import create_dir

print("torch version", torch.__version__)

def myprint(a):
    """Log the print statements"""
    global file
    print(a); file.write(a); file.write("\n"); file.flush()


def student_loss(args, s_logit, t_logit, return_t_logits=False):
    """Kl/ L1 Loss for student"""
    print_logits =  False
    if args.loss == "l1":
        loss_fn = F.l1_loss
        loss = loss_fn(s_logit, t_logit.detach())
    elif args.loss == "kl":
        loss_fn = F.kl_div
        s_logit = F.log_softmax(s_logit, dim=1)
        t_logit = F.softmax(t_logit, dim=1)#here has t-logits!
        loss = loss_fn(s_logit, t_logit.detach(), reduction="batchmean")
    else:
        raise ValueError(args.loss)

    if return_t_logits:
        return loss, t_logit.detach()
    else:
        return loss

def dfme_train(args, teacher, student, generator, device, optimizer, epoch):
    """Main Loop for one epoch of Training Generator and Student"""
    global file
    teacher.eval()
    student.train()
    for param in student.conv1.parameters():
        param.requires_grad = False
    for param in student.conv2.parameters():
        param.requires_grad = False
    
    optimizer_S,  optimizer_G = optimizer

    gradients = []
    correct = 0
    total = 0
    for i in range(args.epoch_itrs):
        """Repeat epoch_itrs times per epoch"""
        for _ in range(args.g_iter):
            #Sample Random Noise
            z = torch.randn((args.batch_size, main_args.nz)).to(device)
            optimizer_G.zero_grad()
            generator.train()
            #Get fake image from generator
            fake = generator(z, pre_x=args.approx_grad) # pre_x returns the output of G before applying the activation


            ## APPOX GRADIENT
            approx_grad_wrt_x, loss_G = estimate_gradient_objective(args, teacher, student, fake, 
                                                epsilon=args.grad_epsilon, m=args.grad_m, num_classes=args.num_classes,
                                                device=device, pre_x=True)

            fake.backward(approx_grad_wrt_x)
                
            optimizer_G.step()

            if i == 0 and args.rec_grad_norm:
                x_true_grad = measure_true_grad_norm(args, fake)
        x_g = []
        y_g = []
        for _ in range(args.d_iter):
            z = torch.randn((args.batch_size, main_args.nz)).to(device)
            fake = generator(z).detach()
            optimizer_S.zero_grad()

            with torch.no_grad(): 
                t_logit = teacher(fake)
            # Correction for the fake logits
            if args.loss == "l1" and args.no_logits:
                t_logit = F.log_softmax(t_logit, dim=1).detach()
                if args.logit_correction == 'min':
                    t_logit -= t_logit.min(dim=1).values.view(-1, 1).detach()
                elif args.logit_correction == 'mean':
                    t_logit -= t_logit.mean(dim=1).view(-1, 1).detach()

            y_g.append(t_logit)
            x_g.append(fake)
            s_logit = student(fake)

            loss_S = student_loss(args, s_logit, t_logit)
            loss_S.backward()
            optimizer_S.step()
            total += len(s_logit)
            correct += torch.argmax(s_logit, dim=1, keepdim=True).eq(torch.argmax(t_logit, dim=1, keepdim=True)).sum().item()
            train_acc = correct / total * 100.
        # Log Results
        if i % args.log_interval == 0:
            myprint(f'Train Epoch: {epoch}[{i}/{args.epoch_itrs}'
                    f' ({100*float(i)/float(args.epoch_itrs):.0f}%)]\tG_Loss: {loss_G.item():.6f} S_loss: {loss_S.item():.6f} ACC: '
                    f'{train_acc:.4f}%')
            """
            myprint(f'Train Epoch: {epoch} [{i}/{args.epoch_itrs}'
                    f' ({100 * float(i) / float(args.epoch_itrs):.0f}%)]\tG_Loss: {loss_G.item():.6f} S_loss: {loss_S.item():.6f} ACC: '
                    f'{train_acc:.4f}%')
            """

            # if args.rec_grad_norm and i == 0:
            #     G_grad_norm, S_grad_norm = compute_grad_norms(generator, student)
            #     if i == 0:
            #         with open(args.log_dir + "/norm_grad.csv", "a") as f:
            #             f.write("%d,%f,%f,%f\n"%(epoch, G_grad_norm, S_grad_norm, x_true_grad))

        # update query budget
        args.query_budget -= args.cost_per_iteration

        if args.query_budget < args.cost_per_iteration:
            return x_g, y_g, loss_S, train_acc
    return x_g, y_g, loss_S, train_acc

def dfme_test(student=None, generator=None, device="cuda", test_loader=None, blackbox=None):
    global file
    student.eval()
    generator.eval()

    test_loss = 0
    correct = 0
    equal_item = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = student(data)

            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            t_pred = blackbox(data)
            t_pred = t_pred.argmax(dim=1, keepdim=True)
            equal_item += pred.eq(t_pred).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    fidelity = equal_item / len(test_loader.dataset) * 100.
    myprint('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%), Fidelity:{}/{} ({:4}/%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy, equal_item, len(test_loader.dataset), fidelity))

    return accuracy, fidelity

def compute_grad_norms(generator, student):
    G_grad = []
    for n, p in generator.named_parameters():
        if "weight" in n:
            # print('===========\ngradient{}\n----------\n{}'.format(n, p.grad.norm().to("cpu")))
            G_grad.append(p.grad.norm().to("cpu"))

    S_grad = []
    for n, p in student.named_parameters():
        if "weight" in n:
            # print('===========\ngradient{}\n----------\n{}'.format(n, p.grad.norm().to("cpu")))
            S_grad.append(p.grad.norm().to("cpu"))
    return  np.mean(G_grad), np.mean(S_grad)


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

def main_runner(main_args):
    ini = main_args.ini
    # Training settings
    parser = argparse.ArgumentParser(description='DFAD CIFAR')
    parser.add_argument('--batch_size', type=int, default=cfg.batch_size, metavar='N',help='input batch size for training (default: 256)')
    parser.add_argument('--query_budget', type=float, default=main_args.query_budget, metavar='N', help='Query budget for the extraction attack in millions (default: 20M)')
    parser.add_argument('--epoch_itrs', type=int, default=50)  
    parser.add_argument('--g_iter', type=int, default=1, help = "Number of generator iterations per epoch_iter")
    parser.add_argument('--d_iter', type=int, default=5, help = "Number of discriminator iterations per epoch_iter")

    parser.add_argument('--lr_S', type=float, default=0.1, metavar='LR', help='Student learning rate (default: 0.1)')
    parser.add_argument('--lr_G', type=float, default=1e-4, help='Generator learning rate (default: 0.1)')

    parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    
    parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'kl'],)
    parser.add_argument('--scheduler', type=str, default='multistep', choices=['multistep', 'cosine', "none"],)
    parser.add_argument('--steps', nargs='+', default = [0.1, 0.3, 0.5], type=float, help = "Percentage epochs at which to take next step")
    parser.add_argument('--scale', type=float, default=3e-1, help = "Fractional decrease in lr")

    #parser.add_argument('--dataset', type=str, default='mnist', choices=['svhn','cifar10', 'mnist'], help='dataset name (default: cifar10)')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--model_arch', type=str, default=cfg.attack_model_arch, choices=classifiers, help='Target model name (default: resnet34_8x)')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--over_factor', type=int, default=cfg.over_factor)
    parser.add_argument('--dataset', type=str, default=cfg.test_dataset)

    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=random.randint(0, 100000), metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--ckpt', type=str, default=cfg.VICTIM_DIR)

    parser.add_argument('--student_load_path', type=str, default=None)
    parser.add_argument('--model_id', type=str, default="debug")

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default=cfg.attack_model_dir)

    # Gradient approximation parameters
    parser.add_argument('--approx_grad', type=int, default=1, help = 'Always set to 1')
    parser.add_argument('--grad_m', type=int, default=1, help='Number of steps to approximate the gradients')#1
    parser.add_argument('--grad_epsilon', type=float, default=1e-3) 
    

    parser.add_argument('--forward_differences', type=int, default=1, help='Always set to 1')

    # Eigenvalues computation parameters
    parser.add_argument('--no_logits', type=int, default=1)
    parser.add_argument('--logit_correction', type=str, default='mean', choices=['none', 'mean'])
    parser.add_argument('--rec_grad_norm', type=int, default=1)
    parser.add_argument('--store_checkpoints', type=int, default=1)
    parser.add_argument('--student_model', type=str, default=cfg.attack_model_arch,
                        help='Student model architecture (default: resnet18_8x)')

    args = parser.parse_args()
    params = vars(args)
    if main_args.MAZE:

        print("\n"*2)
        print("#### /!\ OVERWRITING ALL PARAMETERS FOR MAZE REPLCIATION ####")
        print("\n"*2)
        args.scheduer = "cosine"
        args.loss = "kl"
        # args.batch_size = 32#128
        args.g_iter = 1#1
        args.d_iter = 5#5
        args.grad_m = 10#10
        args.lr_G = 1e-4
        args.lr_S = 1e-1#1e-1

    args.query_budget *= 10**6
    args.query_budget = int(args.query_budget)
    nc = main_args.nc#1 #channel
    img_size = main_args.img_size#28 #size

    out_path = osp.join(cfg.attack_model_dir, cfg.queryset)
    #pprint(args, width=80)
    print(args.log_dir)
    os.makedirs(args.log_dir, exist_ok=True)

    # Save JSON with parameters
    create_dir(out_path)
    params_out_path = osp.join(out_path, 'params.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Prepare the environment
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda:%d"%args.device if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    # Preparing checkpoints for the best Student
    global file

    model_dir = cfg.attack_model_dir+f"checkpoint/student_{args.model_id}"; args.model_dir = model_dir
    if(not os.path.exists(model_dir)):
        os.makedirs(model_dir)
    with open(f"{model_dir}/model_info.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)  
    file = open(f"{args.model_dir}/logs.txt", "w") 

    print(args)
    args.device = device
    args.normalization_coefs = None
    args.G_activation = torch.tanh

    test_dataset_name = cfg.test_dataset
    test_modelfamily = datasets.dataset_to_modelfamily[test_dataset_name]
    test_transform = datasets.modelfamily_to_transforms[test_modelfamily]['test']
    test_dataset = datasets.__dict__[test_dataset_name]
    testset = test_dataset(train=False, transform=test_transform)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    blackbox_dir = cfg.VICTIM_DIR
    teacher, num_classes = Blackbox.from_modeldir_split(blackbox_dir, device)  #probs
    teacher = teacher.get_model()
    teacher.eval()

    attack_model_name = cfg.attack_model_arch
    student = zoo.get_net(attack_model_name, test_modelfamily, pretrained=None,
                          over_factor=cfg.over_factor, num_classes=num_classes)#over_factor=cfg.over_factor,
    print("student,", student)
    student.conv1 = copy.deepcopy(teacher.conv1)
    student.conv2 = copy.deepcopy(teacher.conv2)

    args.num_classes = num_classes

    if ini == True:
        N_ini_query = main_args.N_ini_query
        """
        w1_dir1 = torch.load(f'./mnist_w1/w1_dir-0-{N_ini_query}.pt')
        w1_dir2 = torch.load(f'./mnist_w1/w1_dir-1-{N_ini_query}.pt')
        w1_dir3 = torch.load(f'./mnist_w1/w1_dir-2-{N_ini_query}.pt')
        w1_dir4 = torch.load(f'./mnist_w1/w1_dir-3-{N_ini_query}.pt')
        w1_dir5 = torch.load(f'./mnist_w1/w1_dir-4-{N_ini_query}.pt')
        
        w1_dir1 = torch.load(f'./w1_svhn/w1_dir-0-{N_ini_query}.pt')
        w1_dir2 = torch.load(f'./w1_svhn/w1_dir-1-{N_ini_query}.pt')
        w1_dir3 = torch.load(f'./w1_svhn/w1_dir-2-{N_ini_query}.pt')
        w1_dir4 = torch.load(f'./w1_svhn/w1_dir-3-{N_ini_query}.pt')
        w1_dir5 = torch.load(f'./w1_svhn/w1_dir-4-{N_ini_query}.pt')
        """
        # w1_dir1 = torch.load(f'./data_ini/cifar10/w1_cifar-0-{N_ini_query}.pt')
        # w1_dir2 = torch.load(f'./data_ini/cifar10/w1_cifar-1-{N_ini_query}.pt')
        # w1_dir3 = torch.load(f'./data_ini/cifar10/w1_cifar-2-{N_ini_query}.pt')
        # w1_dir4 = torch.load(f'./data_ini/cifar10/w1_cifar-3-{N_ini_query}.pt')
        # w1_dir5 = torch.load(f'./data_ini/cifar10/w1_cifar-4-{N_ini_query}.pt')

        # w1_dir1 = torch.load(f'./data_ini/cifar100/w1_cifar-0-{N_ini_query}.pt')
        # w1_dir2 = torch.load(f'./data_ini/cifar100/w1_cifar-1-{N_ini_query}.pt')
        # w1_dir3 = torch.load(f'./data_ini/cifar100/w1_cifar-2-{N_ini_query}.pt')
        # w1_dir4 = torch.load(f'./data_ini/cifar100/w1_cifar-3-{N_ini_query}.pt')
        # w1_dir5 = torch.load(f'./data_ini/cifar100/w1_cifar-4-{N_ini_query}.pt')

        w1_dir1 = torch.load(f'./data_ini/mnist_tl/w_mnist_tl0-0-{N_ini_query}.pt')
        w1_dir2= torch.load(f'./data_ini/mnist_tl/w_mnist_tl0-1-{N_ini_query}.pt')
        w1_dir3 = torch.load(f'./data_ini/mnist_tl/w_mnist_tl0-2-{N_ini_query}.pt')
        w1_dir4 = torch.load(f'./data_ini/mnist_tl/w_mnist_tl0-3-{N_ini_query}.pt')
        w1_dir5 = torch.load(f'./data_ini/mnist_tl/w_mnist_tl0-4-{N_ini_query}.pt')

        init_w1 = over_initial(w1_dir1, w1_dir2, w1_dir3, w1_dir4, w1_dir5)
        # student.features[0].weight = Parameter(init_w1)
        # student.first_conv.conv.weight = Parameter(init_w1)
        student.fc1.weight = Parameter(init_w1)

    print(f"\n\t\tTraining with {cfg.victim_model_arch} as a Target\n")
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = teacher(data)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTeacher - Test set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct, len(test_loader.dataset),accuracy))

    #generator = network.gan.GeneratorA(nz=main_args.nz, nc=3, img_size=32, activation=args.G_activation)
    generator = network.gan.GeneratorA(nz=main_args.nz, nc=nc, img_size=img_size, activation=args.G_activation)

    student = student.to(device)
    generator = generator.to(device)

    args.generator = generator
    args.student = student
    args.teacher = teacher

    if args.student_load_path:
        student.load_state_dict(torch.load(args.student_load_path))
        myprint("Student initialized from %s"%(args.student_load_path))
        acc, fidelity = dfme_test(student=student, generator=generator, device = device, test_loader = test_loader, blackbox=teacher)

    ## Compute the number of epochs with the given query budget:
    args.cost_per_iteration = args.batch_size * (args.g_iter * (args.grad_m+1) + args.d_iter)

    number_epochs = args.query_budget // (args.cost_per_iteration * args.epoch_itrs) + 1

    print (f"\nTotal budget: {args.query_budget//1000}k")
    print ("Cost per iterations: ", args.cost_per_iteration)
    print ("Total number of epochs: ", number_epochs)
    for param in student.conv1.parameters():
        param.requires_grad = False
    for param in student.conv2.parameters():
        param.requires_grad = False

    optimizer_S = optim.SGD(filter(lambda p: p.requires_grad, student.parameters()), lr=args.lr_S,
                            weight_decay=args.weight_decay, momentum=0.9)

    # if main_args.MAZE:
    #     optimizer_G = optim.SGD(generator.parameters(), lr=args.lr_G, weight_decay=args.weight_decay, momentum=0.9)
    # else:
    #     optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_G)
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_G)

    steps = sorted([int(step * number_epochs) for step in args.steps])

    print("Learning rate scheduling at steps: ", steps)
    print()

    # if args.scheduler == "multistep":
    #     scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_S, steps, args.scale)
    #     scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, steps, args.scale)
    # elif args.scheduler == "cosine":
    #     scheduler_S = optim.lr_scheduler.CosineAnnealingLR(optimizer_S, number_epochs)
    #     scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, number_epochs)
    scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_S, steps, args.scale)
    scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, steps, args.scale)

    best_test_acc = 0.
    best_fidelity = 0.
    log_path = osp.join(out_path, f'{cfg.attack_set}.log.tsv')
    if not osp.exists(log_path):
        with open(log_path, 'w') as wf:
            columns = ['loss', 'epochs', 'query_number', 'training_acc', 'test_acc@1',
                       'best_test_acc', 'fidelity@1', 'best_fidelilty']
            wf.write('\t'.join(columns) + '\n')
    x_batch = []
    y_batch = []
    for epoch in range(1, number_epochs + 1):
        # Train
        if args.scheduler != "none":
            scheduler_S.step()
            scheduler_G.step()

        x_g, y_g, train_loss, train_acc = dfme_train(args, teacher=teacher, student=student, generator=generator, device=device,
              optimizer=[optimizer_S, optimizer_G], epoch=epoch)
        x_g_t = torch.vstack(x_g)
        y_g_t = torch.vstack(y_g)
        x_batch.append(x_g_t)
        y_batch.append(y_g_t)
        #train_loss, train_acc = recursive_train(args, teacher=teacher, student=student, generator=generator,
        #                                        device=device, optimizer=[optimizer_S, optimizer_G], epoch=epoch)

        # Test
        test_acc, test_fidelity = dfme_test(student=student, generator=generator, device=device,
                             test_loader=test_loader,  blackbox=teacher)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        if test_fidelity >= best_fidelity:
            best_fidelity = test_fidelity
            name = cfg.attack_set
            torch.save(student.state_dict(), out_path + f"/{name}.pth.tar")
            torch.save(generator.state_dict(), out_path + f"/{name}-generator.pth.tar")
            state = {
                'epoch': 100,
                'arch': student.__class__,
                'state_dict': student.state_dict(),
                'best_acc': test_acc,
                'optimizer': optimizer_S,
                'created_on': str(datetime.now()),
            }
            torch.save(state, out_path+
                                  f'/checkpoint_{cfg.attack_set}_{str((args.cost_per_iteration * args.epoch_itrs)*epoch)}.pth.tar')
        if epoch % 10 == 0:
            with open(log_path, 'a') as af:
                train_cols = [train_loss.item(), epoch, (args.cost_per_iteration * args.epoch_itrs)*epoch,
                              train_acc, test_acc,
                              best_test_acc, test_fidelity, best_fidelity]
                af.write('\t'.join([str(c) for c in train_cols]) + '\n')

        if epoch % 100 == 0:
            y_batch_t = torch.vstack(y_batch)
            x_batch_t = torch.vstack(x_batch)
            torch.save(x_batch_t, f"./data_dfme/x_batch_t-{cfg.attack_set}_{str((args.cost_per_iteration * args.epoch_itrs) * epoch)}.pt")
            torch.save(y_batch_t, f"./data_dfme/y_batch_t-{cfg.attack_set}_{str((args.cost_per_iteration * args.epoch_itrs) * epoch)}.pt")
            x_batch = []
            y_batch = []
    with open(log_path, 'a') as af:
        train_cols = [train_loss.item(), epoch, (args.cost_per_iteration * args.epoch_itrs) * epoch,
                      train_acc, test_acc,
                      best_test_acc, test_fidelity, best_fidelity]
        af.write('\t'.join([str(c) for c in train_cols]) + '\n')
    myprint("Best Acc=%.6f"%best_test_acc)

    y_batch_t = torch.vstack(y_batch)
    x_batch_t = torch.vstack(x_batch)
    torch.save(x_batch_t,
               f"./data_dfme/x_batch_t-{cfg.attack_set}_{str((args.cost_per_iteration * args.epoch_itrs) * epoch)}.pt")
    torch.save(y_batch_t,
               f"./data_dfme/y_batch_t-{cfg.attack_set}_{str((args.cost_per_iteration * args.epoch_itrs) * epoch)}.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DFAD')
    parser.add_argument('--ini', type=bool, default=False)
    parser.add_argument('--query_budget', type=float, default=20)
    parser.add_argument('--N_ini_query', type=int, default=4000)#3000 #80000 #2000
    parser.add_argument('--nc', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--MAZE', type=bool, default=False)
    parser.add_argument('--nz', type=int, default=100, help="Size of random noise input to generator (256), 100")
    main_args = parser.parse_args()

    cfg.batch_size = 32  # 64

    # cfg.attack_set = 'mnist_tf_dfme_ini'
    # main_args.ini = True #2
    # cfg.over_factor = 5  # 3
    # main_args.MAZE = False  # 4
    # main_runner(main_args)

    # cfg.attack_set = 'mnist_tf_dfme_over'  # 1
    # main_args.ini = False  # 2
    # cfg.over_factor = 5  # 3
    # main_args.MAZE = False  # 4
    # main_runner(main_args)

    cfg.attack_set = 'mnist_tf_dfme_base'
    main_args.ini = False
    cfg.over_factor = 1 #3
    main_args.MAZE = False  # 4
    main_runner(main_args)

    # cfg.batch_size = 256#192
    # cfg.attack_set = 'mnist_tf_maze_base'
    # main_args.ini = False
    # cfg.over_factor = 1  # 3
    # main_args.MAZE = True  # 4
    # main_runner(main_args)
    #
    # cfg.attack_set = 'mnist_tf_maze_over'
    # main_args.ini = False
    # cfg.over_factor = 5
    # main_args.MAZE = True  # 4
    # main_runner(main_args)
    #
    # cfg.attack_set = 'mnist_tf_maze_ini'
    # main_args.ini = True
    # cfg.over_factor = 5
    # main_args.MAZE = True  # 4
    # main_runner(main_args)





from __future__ import print_function
import argparse, ipdb, json
import copy
import math
from datetime import datetime
from itertools import chain

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
# import network
# from dataloader import get_dataloader
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
from ensemble import Ensemble
from model_scheduler import soft_cross_entropy
from models import get_maze_model
import os.path as osp
import csv

from utils.utils import create_dir

print("torch version", torch.__version__)


def myprint(a):
    """Log the print statements"""
    global file
    print(a);
    file.write(a);
    file.write("\n");
    file.flush()


def student_loss(args, s_logit, t_logit, return_t_logits=False, mis_logit=None):
    """Kl/ L1 Loss for student"""
    print_logits = False
    if args.loss == "l1":
        # t_logit = method_rand_perturbation(teacher_pred=t_logit)
        # t_logit = method_adaptive_misinformation(t_logit, misinformation_pred=mis_logit)
        loss_fn = F.l1_loss
        loss = loss_fn(s_logit, t_logit.detach())
    elif args.loss == "kl":
        loss_fn = F.kl_div
        s_logit = F.log_softmax(s_logit, dim=1)
        t_logit = F.softmax(t_logit, dim=1)
        # t_logit = method_rand_perturbation(teacher_pred=t_logit)
        # t_logit = method_adaptive_misinformation(t_logit, misinformation_pred=mis_logit)
        loss = loss_fn(s_logit, t_logit.detach(), reduction="batchmean")
    else:
        raise ValueError(args.loss)

    if return_t_logits:
        return loss, t_logit.detach()
    else:
        return loss


def inner_product(x1, x2):
    return (x1 * x2).sum()


def cosine(x1, x2):
    norm1 = x1.view(-1).norm(p=2)
    norm2 = x2.view(-1).norm(p=2)
    return (x1 * x2).sum() / (norm1 * norm2)


def get_Gty(bx, student, y, create_graph=False):
    """
    computes G^T y, where y is the posterior used in cross-entropy loss

    Works with student_pred of shape (N, K)

    :param bx: a batch of image inputs (CIFAR or ImageNet in our experiments)
    :param student: student network
    :param y: posterior to be used for cross-entropy loss
    :create_graph: whether to create the graph for double backprop
    :returns: G^T y, a 1D tensor with length equal to the number of params in backprop_modules
    """
    logits = student(bx)
    normalized_logits = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    # cross-entropy loss
    loss = (y * normalized_logits).sum(1).mean(0)
    student.zero_grad()
    student_grad = torch.autograd.grad([loss],
                                       student.parameters(),
                                       create_graph=create_graph,
                                       retain_graph=False,
                                       only_inputs=True)
    Gty = torch.cat([x.view(-1) for x in student_grad])

    return Gty


def make_param_generator(modules):
    """
    :param modules: a list of PyTorch modules
    :returns: a generator that chains together all the individual parameter generators
    """
    return chain(*[module.parameters() for module in modules])


def get_Gty_partial(bx, student, y, backprop_modules, create_graph=False):
    """
    Same as get_Gty, but only returns gradient wrt the provided parameters specified in backprop_modules.

    This is a component of the MAD defense specified in the official MAD GitHub repository. It speeds up
    their method considerably by only backpropping through part of the network.
    We didn't explore this functionality with GRAD^2, since it is already fast enough,
    but it could be used to make things even faster.

    :param bx: a batch of image inputs (CIFAR or ImageNet in our experiments)
    :param student: student network
    :param y: posterior to be used for cross-entropy loss
    :backprop_modules: modules of the student network to backprop through
    :create_graph: whether to create the graph for double backprop
    :returns: G^T y, a 1D tensor with length equal to the number of params in backprop_modules
    """
    for p in student.parameters():
        p.requires_grad = False
    for p in make_param_generator(backprop_modules):
        p.requires_grad = True

    logits = student(bx)
    normalized_logits = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    # cross-entropy loss
    loss = (y * normalized_logits).sum(1).mean(0)
    student.zero_grad()
    student_grad = torch.autograd.grad([loss],
                                       make_param_generator(backprop_modules),
                                       create_graph=create_graph,
                                       retain_graph=False,
                                       only_inputs=True)
    Gty = torch.cat([x.view(-1) for x in student_grad])

    for p in student.parameters():
        p.requires_grad = True

    return Gty


def method_gradient_redirection(bx, teacher, students, epsilon=0.1, backprop_modules=None, override_grad=False):
    """
    Find perturbation minimizing inner product of perturbed gradient with original gradient.
    The perturbation is constrained to have an L1 norm of epsilon and to be such that the
    perturbed posterior is on the simplex.

    Optionally uses an ensemble of students (not used in the paper, but included for completeness)

    NOTE: The paper describes gradient redirection as maximizing the inner product, while this implementation minimizes inner product. They are functionally
    identical, but this means that to maximize inner product with the all-ones vector you should pass in override_grad=-1*torch.ones(...)

    NOTE: This function with the appropriate surrogate (i.e., student) model is the GRAD^2 defense from the paper.

    :param bx: input batch of tensors for teacher and student networks
    :param teacher: teacher network, outputs logits
    :param students: list of student networks
    :param epsilons: list of target epsilons (L1 distance)
    :param backprop_modules: optional list (or list of lists) of PyTorch modules from the student networks, specifying the parameters to target with the defense
                             (NOTE: This is not used in the GRAD^2 defense in the paper and is included for completeness.)
    :param override_grad: optional tensor specifying the target direction for gradient redirection; this is how the all-ones and watermark experiments are specified
    :returns: perturbation satisfying L1 constraint
    """
    # FIRST, GET DIRECTION OF PERTURBATION

    # get Gty_tilde and Gty for each student network
    with torch.no_grad():
        teacher_logits = teacher(bx)
        teacher_pred = torch.softmax(teacher_logits, dim=1)
    teacher_pred.requires_grad_()

    all_Gty_tilde = []
    all_Gty = []
    for student in students:
        student.zero_grad()
        y_tilde = teacher_pred.clone()  # initialize y_tilde to the teacher posterior
        if backprop_modules is None:
            Gty_tilde = get_Gty(bx, student, y_tilde, create_graph=True)
        else:
            Gty_tilde = get_Gty_partial(bx, student, y_tilde, backprop_modules, create_graph=True)
        Gty = Gty_tilde.detach()  # this works if we initialize y_tilde = y, as we do here
        all_Gty_tilde.append(Gty_tilde)
        all_Gty.append(Gty)

    Gty_tilde = torch.cat(all_Gty_tilde)
    Gty = torch.cat(all_Gty)

    if override_grad is not False:  # "is" matters here; cannot use ==
        Gty = override_grad

    # now compute the objective and double backprop
    objective = inner_product(Gty_tilde, Gty)
    grad_pred_inner_prod = torch.autograd.grad([objective], [teacher_pred], only_inputs=True)[0]

    # now compute the optimal perturbation using our algorithm, separately for each example in the batch
    all_teacher_pred_perturbed = []
    for idx in range(bx.shape[0]):
        # teacher_pred_perturbed_per_epsilon = []

        epsilon_target = epsilon
        # algorithm start
        c = torch.argsort(grad_pred_inner_prod[idx])  # smallest to largest
        take_pointer = len(c) - 1  # where to take probability mass from; for L1 constraint, we always give to c[0]

        with torch.no_grad():
            tmp = teacher_pred.clone()[idx]
            can_give = min(1 - tmp[c[0]].item(), epsilon_target / 2)
            found_to_give = torch.zeros(1).cuda()[0]
            while (found_to_give < can_give):
                found_here = tmp[c[take_pointer]].item()
                # print('found_to_give: {} \tcan_give: {} \tfound_here: {}'.format(found_to_give, can_give, found_here))
                if found_to_give + found_here <= can_give:
                    tmp[c[take_pointer]] -= found_here
                    found_to_give += found_here
                elif found_to_give + found_here > can_give:
                    # print('got here')
                    tmp[c[take_pointer]] -= can_give - found_to_give
                    found_to_give += can_give - found_to_give
                take_pointer -= 1
                if np.isclose(found_to_give.item(), can_give):
                    break
            tmp[c[0]] += found_to_give

            # to handle arithmetic errors (very minor when they occur)
            tmp = tmp.cuda()
            teacher_pred_perturbed = torch.softmax(torch.log(torch.clamp(tmp, 1e-15, 1)), dim=0).unsqueeze(0)
        # algorithm end
        # teacher_pred_perturbed_per_epsilon.append(teacher_pred_perturbed)

        # teacher_pred_perturbed_per_epsilon = torch.cat(teacher_pred_perturbed_per_epsilon, dim=0)
        # all_teacher_pred_perturbed.append(teacher_pred_perturbed_per_epsilon.unsqueeze(0))

    # teacher_pred_perturbed = torch.cat(all_teacher_pred_perturbed, dim=0)

    return teacher_pred_perturbed.detach()


def method_rand_perturbation(teacher_pred, epsilon=0.5, num_classes=10):
    """
    :param bx: input batch of tensors for teacher and student networks
    :param teacher: teacher network, outputs logits
    :param student: unused
    :param epsilons: epsilons for perturbation
    :returns: random perturbation of teacher posterior satisfying L1 constraint
    """
    # for y
    max_indices = torch.max(teacher_pred, dim=1)[1]
    perturbation_target = torch.zeros_like(teacher_pred)
    for i in range(len(max_indices)):
        choices = list(range(num_classes))
        choices.remove(max_indices[i].int().item())
        assert len(choices) == num_classes - 1, 'error'
        loc = np.random.choice(choices)
        perturbation_target[i, loc] = 1

    perturbed_pred = (epsilon / 2) * perturbation_target + (1 - (epsilon / 2)) * teacher_pred

    return perturbed_pred


def method_adaptive_misinformation(teacher_pred, student=None, epsilon=0.5, oe_model=None, misinformation_pred=None):
    """
    :param bx: input batch of tensors for teacher and student networks
    :param teacher: teacher network, outputs logits
    :param student: list of students (None for reverse sigmoid)
    :param epsilons: a list of tau parameters in the adaptive misinformation method (named epsilon for convenience)
    :param oe_model: the outlier exposure model to use for detecting OOD data; if None, then use the MSP of the teacher
    :param misinformation_model: the misinformation model in the adaptive misinformation defense
    :returns: perturbed posterior
    """
    assert misinformation_model is not None, 'must supply a misinformation model'

    tau = epsilon  # renaming for clarity
    nu = 5

    # get clean posterior
    # with torch.no_grad():
    #     teacher_logits = teacher(bx)
    #     teacher_pred = torch.softmax(teacher_logits, dim=1)
    #
    # # get misinformation posterior
    # with torch.no_grad():
    #     misinformation_logits = misinformation_model(bx)
    #     misinformation_pred = torch.softmax(misinformation_logits, dim=1)

    # get inlier scores
    # if oe_model is not None:
    #     with torch.no_grad():
    #         oe_logits = oe_model(bx)
    #         inlier_scores = torch.softmax(oe_logits, dim=1).max(dim=1)[0]
    # else:
    inlier_scores = teacher_pred.max(dim=1)[0]

    all_outs = []
    # for each beta, perturb the posterior with the adaptive misinformation method
    alpha = torch.reciprocal(1 + torch.exp(nu * (inlier_scores - tau))).unsqueeze(-1)
    out = (1 - alpha) * teacher_pred + alpha * misinformation_pred

    return out


# def train(args, teacher, student, generator, device, optimizer, epoch):
#     """Main Loop for one epoch of Training Generator and Student"""
#     global file
#     teacher.eval()
#     student.train()
#
#     optimizer_S,  optimizer_G = optimizer
#
#     # gradients = []
#     correct = 0
#     total = 0
#     for i in range(args.epoch_itrs):
#         """Repeat epoch_itrs times per epoch"""
#         for _ in range(args.g_iter):
#             #Sample Random Noise
#             z = torch.randn((args.batch_size, main_args.nz)).cuda()
#             optimizer_G.zero_grad()
#             generator.train()
#             #Get fake image from generator
#             fake = generator(z, pre_x=args.approx_grad) # pre_x returns the output of G before applying the activation
#             ## APPOX GRADIENT
#             approx_grad_wrt_x, loss_G = estimate_gradient_objective(args, teacher, student, fake,
#                                                 epsilon = args.grad_epsilon, m = args.grad_m, num_classes=args.num_classes,
#                                                 device=device, pre_x=True)
#
#             fake.backward(approx_grad_wrt_x)
#
#             optimizer_G.step()
#
#             if i == 0 and args.rec_grad_norm:
#                 x_true_grad = measure_true_grad_norm(args, fake)
#         x_g = []
#         y_g = []
#         for _ in range(args.d_iter):
#             z = torch.randn((args.batch_size, main_args.nz)).cuda()
#             fake = generator(z).detach()
#             optimizer_S.zero_grad()
#
#             with torch.no_grad():
#                 t_logit = teacher(fake)
#             # Correction for the fake logits
#             if args.loss == "l1" and args.no_logits:#DEME
#                 t_logit = F.log_softmax(t_logit, dim=1).detach()
#                 if args.logit_correction == 'min':
#                     t_logit -= t_logit.min(dim=1).values.view(-1, 1).detach()
#                 elif args.logit_correction == 'mean':
#                     t_logit -= t_logit.mean(dim=1).view(-1, 1).detach()
#
#             # with torch.no_grad():
#             #     mis_logit = misinformation_model(fake)
#             # # Correction for the fake logits
#             # if args.loss == "l1" and args.no_logits:#DEME
#             #     mis_logit = F.log_softmax(mis_logit, dim=1).detach()
#             #     if args.logit_correction == 'min':
#             #         mis_logit -= mis_logit.min(dim=1).values.view(-1, 1).detach()
#             #     elif args.logit_correction == 'mean':
#             #         mis_logit -= mis_logit.mean(dim=1).view(-1, 1).detach()
#
#             y_g.append(t_logit)
#             x_g.append(fake)
#             s_logit = student(fake)
#
#             loss_S = student_loss(args, s_logit, t_logit) #, mis_logit=mis_logit
#             loss_S.backward()
#             optimizer_S.step()
#             total += len(s_logit)
#             correct += torch.argmax(s_logit, dim=1, keepdim=True).eq(torch.argmax(t_logit, dim=1, keepdim=True)).sum().item()
#             train_acc = correct / total * 100.
#         # Log Results
#         if i % args.log_interval == 0:
#             myprint(f'Train Epoch: {epoch}[{i}/{args.epoch_itrs}'
#                     f' ({100*float(i)/float(args.epoch_itrs):.0f}%)]\tG_Loss: {loss_G.item():.6f} S_loss: {loss_S.item():.6f} ACC: '
#                     f'{train_acc:.4f}%')
#             """
#             myprint(f'Train Epoch: {epoch} [{i}/{args.epoch_itrs}'
#                     f' ({100 * float(i) / float(args.epoch_itrs):.0f}%)]\tG_Loss: {loss_G.item():.6f} S_loss: {loss_S.item():.6f} ACC: '
#                     f'{train_acc:.4f}%')
#             """
#
#             # if args.rec_grad_norm and i == 0:
#             #     G_grad_norm, S_grad_norm = compute_grad_norms(generator, student)
#             #     if i == 0:
#             #         with open(args.log_dir + "/norm_grad.csv", "a") as f:
#             #             f.write("%d,%f,%f,%f\n"%(epoch, G_grad_norm, S_grad_norm, x_true_grad))
#
#         # update query budget
#         args.query_budget -= args.cost_per_iteration
#
#         if args.query_budget < args.cost_per_iteration:
#             return x_g, y_g, loss_S, train_acc
#     return x_g, y_g, loss_S, train_acc
def disguide_gen_loss(fake, generator, student_ensemble, args):
    """Compute generator loss for DisGUIDE method. Update weights based on loss.
    Calculates weighted average between disagreement loss and diversity loss."""

    preds = []
    for idx in range(student_ensemble.size()):
        preds.append(student_ensemble(fake, idx=idx))  # 2x [batch_size, K] Last dim is logits
    preds = torch.stack(preds, dim=1)                  # [batch_size, 2, K]
    preds = F.softmax(preds, dim=2)                    # [batch_size, 2, K] Last dim is confidence values.
    std = torch.std(preds, dim=1)                      # std has shape [batch_size, K]. standard deviation over models
    loss_G = -torch.mean(std)                          # Disagreement Loss
    if args.lambda_div != 0:
        soft_vote_mean = torch.mean(torch.mean(preds + 0.000001, dim=1),
                                    dim=0)  # [batch_size, 2, K] -> [batch_size, K] -> [K]
        loss_G += args.lambda_div * (torch.sum(soft_vote_mean * torch.log(soft_vote_mean)))  # Diversity Loss
    loss_G.backward()
    return loss_G.item()

def supervised_student_training(student_ensemble, fake, t_logit, optimizer, args):
    """Calculate loss and update weights for students in a supervised fashion"""
    student_iter_preds = []
    student_iter_loss = 0
    for i in range(student_ensemble.size()):
        s_logit = student_ensemble(fake, idx=i)
        with torch.no_grad():
            student_iter_preds.append(F.softmax(s_logit, dim=-1).detach())
        loss_s = student_loss(args, s_logit, t_logit)  # Helper function which handles soft- and hard-label settings
        loss_s.backward()
        student_iter_loss += loss_s.item()
    optimizer.step()
    return torch.stack(student_iter_preds, dim=1), student_iter_loss


def dfme_train(args, teacher, student, generator, device, optimizer, epoch):
    """Main Loop for one epoch of Training Generator and Student"""
    global file
    teacher.eval()
    student.train()

    optimizer_S, optimizer_G = optimizer

    gradients = []
    correct = 0
    total = 0
    for i in range(args.epoch_itrs):
        """Repeat epoch_itrs times per epoch"""
        for _ in range(args.g_iter):
            # Sample Random Noise
            z = torch.randn((args.batch_size, main_args.nz)).cuda()
            optimizer_G.zero_grad()
            generator.train()
            # Get fake image from generator
            fake = generator(z, pre_x=args.approx_grad)  # pre_x returns the output of G before applying the activation

            ## APPOX GRADIENT
            if main_args.attack_type == 'DFME' or main_args.attack_type == 'MAZE':
                 approx_grad_wrt_x, record_loss_G = estimate_gradient_objective(args, teacher, student, fake,
                                                                    epsilon=args.grad_epsilon, m=args.grad_m,
                                                                    num_classes=args.num_classes,
                                                                    device=device, pre_x=True)
                 fake.backward(approx_grad_wrt_x)

            elif main_args.attack_type == 'DISGUIDE':
                record_loss_G = disguide_gen_loss(fake, generator, student, args)

            optimizer_G.step()

            # if i == 0 and args.rec_grad_norm:
            #     x_true_grad = measure_true_grad_norm(args, fake)
        x_g = []
        y_g = []
        for _ in range(args.d_iter):
            z = torch.randn((args.batch_size, main_args.nz)).cuda()
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

            if main_args.attack_type == 'DISGUIDE':
                student_iter_preds, record_loss_S = supervised_student_training(student, fake, t_logit, optimizer_S, args)
                # print('student_iter_preds', student_iter_preds.shape)

                total += len(student_iter_preds)
                correct += torch.argmax(torch.mean(student_iter_preds, dim=1), dim=1, keepdim=True).eq(
                    torch.argmax(t_logit, dim=1, keepdim=True)).sum().item()
            else:
                s_logit = student(fake)
                # print("fake,", fake.shape)
                # print("student,", student)

                loss_S = student_loss(args, s_logit, t_logit)
                loss_S.backward()
                optimizer_S.step()
                total += len(s_logit)
                correct += torch.argmax(s_logit, dim=1, keepdim=True).eq(
                torch.argmax(t_logit, dim=1, keepdim=True)).sum().item()
                record_loss_S = loss_S.item()

            train_acc = correct / total * 100.
        # Log Results
        if i % args.log_interval == 0:
            myprint(f'Train Epoch: {epoch}[{i}/{args.epoch_itrs}'
                    f' ({100 * float(i) / float(args.epoch_itrs):.0f}%)]\tG_Loss: {record_loss_G:.6f} S_loss: {record_loss_S:.6f} ACC: '
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
            return x_g, y_g, record_loss_S, train_acc
    return x_g, y_g, record_loss_S, train_acc

def get_model_preds_and_true_labels(model, loader, device="cuda"):
    """Compute the predictions for a model on a dataset and return this together with the true labels"""
    model.eval()
    targets = []
    preds = []
    with torch.no_grad():
        for (data, target) in loader:
            data, target = data.to(device), target.to(device)
            targets.append(target)
            output = model(data)
            assert len(output.shape) == 2 or (len(output.shape) == 3 and isinstance(model, Ensemble))
            preds.append(F.softmax(output, dim=-1))
    targets = torch.cat(targets, dim=0)
    preds = torch.cat(preds, dim=0)
    return preds, targets

def test(student=None, generator=None, device="cuda", test_loader=None, blackbox=None):
    global file
    student.eval()
    generator.eval()

    test_loss = 0
    correct = 0
    equal_item = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            if main_args.attack_type == 'DISGUIDE':
                output = student(data, idx=0)
            else:
                output = student(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
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
    return np.mean(G_grad), np.mean(S_grad)


def over_initial_cnn(w1_dir1, w1_dir2, w1_dir3, w1_dir4, w1_dir5, channel=3, over_factor=5):
    # over_factor
    d, m1 = w1_dir1.shape#d=3*kernel*kernel,
    w1 = w1_dir1
    w2 = w1_dir2
    w3 = w1_dir3
    w4 = w1_dir4
    w5 = w1_dir5  # d * m1

    # w1= torch.randn_like(w1).cuda()
    # w2 = torch.randn_like(w2).cuda()
    # w3 = torch.randn_like(w1).cuda()
    # w4 = torch.randn_like(w1).cuda()
    # w5 = torch.randn_like(w1).cuda()

    init_w1_1 = ((w1 / torch.norm(w1, dim=1).view(-1, 1)) * math.sqrt(
        2. / d))  # mag1# * w1 #d*k torch.randn([k, d]).cuda()
    init_w1_2 = (w2 / torch.norm(w2, dim=1).view(-1, 1)) * math.sqrt(
        2. / d)  # mag3#torch.randn([k, d]).cuda() * math.sqrt(2/d)
    init_w1_3 = (w3 / torch.norm(w3, dim=1).view(-1, 1)) * math.sqrt(2. / d)
    init_w1_4 = (w4 / torch.norm(w4, dim=1).view(-1, 1)) * math.sqrt(2. / d)  # mag4
    init_w1_5 = (w5 / torch.norm(w5, dim=1).view(-1, 1)) * math.sqrt(2. / d)  # mag5

    d, m1 = w1_dir1.shape
    # print("init_w1_1,", init_w1_1-init_w1_2)
    # print("init_w1_2,", init_w1_2.shape)
    # init_w1_3 = torch.randn_like(w1_dir).cuda()#mag3 * w1
    # m1 * d
    init_w1 = torch.vstack((init_w1_1.T, init_w1_2.T, init_w1_3.T, init_w1_4.T, init_w1_5.T)).view(m1 * 5, channel,
                                                                                                   int(np.sqrt(
                                                                                                       d / channel)),
                                                                                                   int(np.sqrt(
                                                                                                       d / channel)))  # [:int(m1*over_factor)] #init_w1_4.T, init_w1_5.T
    if over_factor>5:
        init_w1 = torch.vstack((init_w1, init_w1))

    index = np.random.randint(0, int(len(init_w1) - 1), int(m1 * over_factor))
    return init_w1[index]

def main_runner(main_args):
    ini = main_args.ini
    # Training settings
    parser = argparse.ArgumentParser(description='DFAD CIFAR')
    parser.add_argument('--batch_size', type=int, default=cfg.batch_size, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--query_budget', type=float, default=main_args.query_budget, metavar='N',
                        help='Query budget for the extraction attack in millions (default: 20M)')
    parser.add_argument('--epoch_itrs', type=int, default=50)
    parser.add_argument('--g_iter', type=int, default=1, help="Number of generator iterations per epoch_iter")
    parser.add_argument('--d_iter', type=int, default=5, help="Number of discriminator iterations per epoch_iter")

    parser.add_argument('--lr_S', type=float, default=0.05, metavar='LR', help='Student learning rate (default: 0.1)')
    parser.add_argument('--lr_G', type=float, default=1e-4, help='Generator learning rate (default: 0.1)')

    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'kl'], )
    parser.add_argument('--scheduler', type=str, default='multistep', choices=['multistep', 'cosine', "none"], )
    parser.add_argument('--steps', nargs='+', default=[0.1, 0.3, 0.5], type=float,
                        help="Percentage epochs at which to take next step")
    parser.add_argument('--scale', type=float, default=3e-1, help="Fractional decrease in lr")

    # parser.add_argument('--dataset', type=str, default='mnist', choices=['svhn','cifar10', 'mnist'], help='dataset name (default: cifar10)')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--model_arch', type=str, default=cfg.attack_model_arch, choices=classifiers,
                        help='Target model name (default: resnet34_8x)')
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
    parser.add_argument('--approx_grad', type=int, default=1, help='Always set to 1')
    parser.add_argument('--grad_m', type=int, default=1, help='Number of steps to approximate the gradients')  # 1
    parser.add_argument('--grad_epsilon', type=float, default=1e-3)

    parser.add_argument('--forward_differences', type=int, default=1, help='Always set to 1')

    # Eigenvalues computation parameters
    parser.add_argument('--no_logits', type=int, default=1)
    parser.add_argument('--logit_correction', type=str, default='mean', choices=['none', 'mean'])
    parser.add_argument('--rec_grad_norm', type=int, default=1)
    parser.add_argument('--store_checkpoints', type=int, default=1)
    parser.add_argument('--student_model', type=str, default=cfg.attack_model_arch,
                        help='Student model architecture (default: resnet18_8x)')
    parser.add_argument('--lambda-div', type=float, default=0,
                        help='Penalty weight for generator class diversity in PV training'
                             'TODO: Connect this to DFMS as well')

    args = parser.parse_args()
    params = vars(args)
    if main_args.attack_type=='MAZE':
        print("\n" * 2)
        print("#### /!\ OVERWRITING ALL PARAMETERS FOR MAZE REPLCIATION ####")
        print("\n" * 2)
        args.scheduer = "cosine"
        args.loss = "kl"
        # args.batch_size = 32#128
        args.g_iter = 1  # 1
        args.d_iter = 5  # 5
        args.grad_m = 10  # 10
        # args.lr_G = 1e-4
        # args.lr_S = 1e-1

    args.query_budget *= 10 ** 6
    args.query_budget = int(args.query_budget)
    nc = main_args.nc  # 1 #channel
    img_size = main_args.img_size  # 28 #size

    out_path = osp.join(cfg.attack_model_dir, cfg.test_dataset)
    # pprint(args, width=80)
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

    device = torch.device("cuda:%d" % args.device if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Preparing checkpoints for the best Student
    global file

    model_dir = cfg.attack_model_dir + f"checkpoint/student_{args.model_id}";
    args.model_dir = model_dir
    if (not os.path.exists(model_dir)):
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
    testset = test_dataset(train=False, transform=test_transform, download=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    blackbox_dir = cfg.VICTIM_DIR
    teacher, num_classes = Blackbox.from_modeldir_split(blackbox_dir)  # probs
    teacher = teacher.get_model()
    teacher.eval()
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        teacher = nn.DataParallel(teacher)
    teacher = teacher.cuda()

    attack_model_name = cfg.attack_model_arch
    # student = zoo.get_net(attack_model_name, test_modelfamily, None, over_factor=cfg.over_factor, num_classes=num_classes)#over_factor=cfg.over_factor,
    student = zoo.get_net(attack_model_name, test_modelfamily, None, over_factor=cfg.over_factor,
                          num_classes=num_classes)
    # student, _ = Blackbox.from_modeldir_split_attack_mode(out_path,
    #                                                       'ordinary-over_rec_checkpoint-0.pth.tar', device)
    # student = student.get_model()

    args.num_classes = num_classes
    print("student_model", student)
    if ini == True:
        N_ini_query = main_args.N_ini_query
        # w1_dir1 = torch.load(f'./data_ini/cifar10/w1_cifar-0-{N_ini_query}.pt').cuda()
        # w1_dir2 = torch.load(f'./data_ini/cifar10/w1_cifar-1-{N_ini_query}.pt').cuda()
        # w1_dir3 = torch.load(f'./data_ini/cifar10/w1_cifar-2-{N_ini_query}.pt').cuda()
        # w1_dir4 = torch.load(f'./data_ini/cifar10/w1_cifar-3-{N_ini_query}.pt').cuda()
        # w1_dir5 = torch.load(f'./data_ini/cifar10/w1_cifar-4-{N_ini_query}.pt').cuda()

        # w1_dir1 = torch.load(f'./data_ini/svhn-prac/w1_0-{N_ini_query}.pt').cuda()
        # w1_dir2 = torch.load(f'./data_ini/svhn-prac/w1_1-{N_ini_query}.pt').cuda()
        # w1_dir3 = torch.load(f'./data_ini/svhn-prac/w1_2-{N_ini_query}.pt').cuda()
        # w1_dir4 = torch.load(f'./data_ini/svhn-prac/w1_3-{N_ini_query}.pt').cuda()
        # w1_dir5 = torch.load(f'./data_ini/svhn-prac/w1_4-{N_ini_query}.pt').cuda()

        # w1_dir1 = torch.load(f'./data_ini/fashionmnist/w1_mnist-0-{N_ini_query}.pt').cuda()
        # w1_dir2 = torch.load(f'./data_ini/fashionmnist/w1_mnist-1-{N_ini_query}.pt').cuda()
        # w1_dir3 = torch.load(f'./data_ini/fashionmnist/w1_mnist-2-{N_ini_query}.pt').cuda()
        # w1_dir4 = torch.load(f'./data_ini/fashionmnist/w1_mnist-3-{N_ini_query}.pt').cuda()
        # w1_dir5 = torch.load(f'./data_ini/fashionmnist/w1_mnist-4-{N_ini_query}.pt').cuda()

        # w1_dir1 = torch.load(f'./data_ini/mnist/w1_mnist-0-{N_ini_query}.pt')
        # w1_dir2 = torch.load(f'./data_ini/mnist/w1_mnist-1-{N_ini_query}.pt')
        # w1_dir3 = torch.load(f'./data_ini/mnist/w1_mnist-2-{N_ini_query}.pt')
        # w1_dir4 = torch.load(f'./data_ini/mnist/w1_mnist-3-{N_ini_query}.pt')
        # w1_dir5 = torch.load(f'./data_ini/mnist/w1_mnist-4-{N_ini_query}.pt')

        w1_dir1 = torch.load(f'./data_ini/svhn/w1_1-0-{N_ini_query}.pt').cuda()
        w1_dir2 = torch.load(f'./data_ini/svhn/w1_1-1-{N_ini_query}.pt').cuda()
        w1_dir3 = torch.load(f'./data_ini/svhn/w1_1-2-{N_ini_query}.pt').cuda()
        w1_dir4 = torch.load(f'./data_ini/svhn/w1_1-3-{N_ini_query}.pt').cuda()
        w1_dir5 = torch.load(f'./data_ini/svhn/w1_1-4-{N_ini_query}.pt').cuda()
        #for vgg
        transferred_over_factor = (cfg.over_factor*16./16)#1.5
        #the maximum = 16*5=80
        print("student.conv1.weight", student.conv1.weight.shape)
        init_w1 = over_initial_cnn(w1_dir1, w1_dir2, w1_dir3, w1_dir4, w1_dir5, channel=nc, over_factor=transferred_over_factor)
        print("init_w1", init_w1.shape)

        student.conv1.weight = Parameter(init_w1) #for lenet-5, densenet
        # student.first_conv.conv.weight = Parameter(init_w1) #for resnet
        # print("student.features[0]", student.features[0])
        # student.features[0].weight = Parameter(init_w1) #for features
        # print("student.features[0]", student.features[0])

    print(f"\n\t\tTraining with {cfg.victim_model_arch} as a Target\n")
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            output = teacher(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTeacher - Test set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct, len(test_loader.dataset), accuracy))

    # generator = network.gan.GeneratorA(nz=main_args.nz, nc=3, img_size=32, activation=args.G_activation)
    generator = network.gan.GeneratorA(nz=main_args.nz, nc=nc, img_size=img_size, activation=args.G_activation)
    # checkpoint_path = osp.join(out_path, 'mnist-dfme_base-generator.pth.tar')
    # print("=> loading checkpoint '{}'".format(checkpoint_path))
    # checkpoint = torch.load(checkpoint_path)
    # generator.load_state_dict(checkpoint)
    if args.student_load_path:
        student.load_state_dict(torch.load(args.student_load_path))
        myprint("Student initialized from %s" % (args.student_load_path))
        acc, fidelity = test(student=student, generator=generator, device=device, test_loader=test_loader,
                             blackbox=teacher)

    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        student = nn.DataParallel(student)
        generator = nn.DataParallel(generator)

    if main_args.attack_type == "DISGUIDE":
        student = Ensemble(student)

    student = student.cuda()
    generator = generator.cuda()

    args.generator = generator
    args.student = student
    args.teacher = teacher

    ## Compute the number of epochs with the given query budget:
    args.cost_per_iteration = args.batch_size * (args.g_iter * (args.grad_m + 1) + args.d_iter)

    number_epochs = args.query_budget // (args.cost_per_iteration * args.epoch_itrs) + 1

    print(f"\nTotal budget: {args.query_budget // 1000}k")
    print("Cost per iterations: ", args.cost_per_iteration)
    print("Total number of epochs: ", number_epochs)

    optimizer_S = optim.SGD(student.parameters(), lr=args.lr_S, weight_decay=args.weight_decay, momentum=0.9)

    if main_args.attack_type == 'MAZE':
        optimizer_G = optim.SGD(generator.parameters(), lr=args.lr_G, weight_decay=args.weight_decay, momentum=0.9)
    else:
        optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_G)
    # for cifar10
    # optimizer_G = optim.SGD(generator.parameters(), lr=args.lr_G, weight_decay=args.weight_decay, momentum=0.9)

    steps = sorted([int(step * number_epochs) for step in args.steps])

    print("Learning rate scheduling at steps: ", steps)
    print()

    if args.scheduler == "multistep":
        scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_S, steps, args.scale)
        scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, steps, args.scale)
    elif args.scheduler == "cosine":
        scheduler_S = optim.lr_scheduler.CosineAnnealingLR(optimizer_S, number_epochs)
        scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, number_epochs)

    # for cifar10
    # scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_S, steps, args.scale)
    # scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, steps, args.scale)

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

        x_g, y_g, train_loss, train_acc = dfme_train(args, teacher=teacher, student=student, generator=generator,
                                                     device=device,
                                                     optimizer=[optimizer_S, optimizer_G], epoch=epoch)
        x_g_t = torch.vstack(x_g)
        y_g_t = torch.vstack(y_g)
        x_batch.append(x_g_t)
        y_batch.append(y_g_t)
        # train_loss, train_acc = recursive_train(args, teacher=teacher, student=student, generator=generator,
        #                                        device=device, optimizer=[optimizer_S, optimizer_G], epoch=epoch)

        # Test
        with torch.no_grad():
            if main_args.attack_type == 'DISGUIDE':
                test_acc, test_fidelity = test(student=student, generator=generator, device=device,
                                               test_loader=test_loader, blackbox=teacher)
            else:
                test_acc, test_fidelity = test(student=student, generator=generator, device=device,
                                           test_loader=test_loader, blackbox=teacher)
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
                torch.save(state, out_path +
                           f'/checkpoint_{cfg.attack_set}_{str((args.cost_per_iteration * args.epoch_itrs) * epoch)}.pth.tar')

        if epoch % 10 == 0:
            with open(log_path, 'a') as af:
                train_cols = [train_loss, epoch, (args.cost_per_iteration * args.epoch_itrs) * epoch,
                              train_acc, test_acc,
                              best_test_acc, test_fidelity, best_fidelity]
                af.write('\t'.join([str(c) for c in train_cols]) + '\n')
        if epoch % 50 == 0:
            y_batch_t = torch.vstack(y_batch)
            x_batch_t = torch.vstack(x_batch)
            torch.save(x_batch_t,
                       f"./data_dfme/x_batch_t-{cfg.attack_set}_{str((args.cost_per_iteration * args.epoch_itrs) * epoch)}.pt")
            torch.save(y_batch_t,
                       f"./data_dfme/y_batch_t-{cfg.attack_set}_{str((args.cost_per_iteration * args.epoch_itrs) * epoch)}.pt")
            x_batch = []
            y_batch = []

        torch.cuda.empty_cache()
    with open(log_path, 'a') as af:
        train_cols = [train_loss, epoch, (args.cost_per_iteration * args.epoch_itrs) * epoch,
                      train_acc, test_acc,
                      best_test_acc, test_fidelity, best_fidelity]
        af.write('\t'.join([str(c) for c in train_cols]) + '\n')
    myprint("Best Acc=%.6f" % best_test_acc)

    y_batch_t = torch.vstack(y_batch)
    x_batch_t = torch.vstack(x_batch)
    torch.save(x_batch_t,
               f"./data_dfme/x_batch_t-{cfg.attack_set}_{str((args.cost_per_iteration * args.epoch_itrs) * epoch)}.pt")
    torch.save(y_batch_t,
               f"./data_dfme/y_batch_t-{cfg.attack_set}_{str((args.cost_per_iteration * args.epoch_itrs) * epoch)}.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DFAD')
    parser.add_argument('--ini', type=bool, default=False)
    parser.add_argument('--query_budget', type=float, default=10, help='10M for CIFAR10-dfme, 20M for CIFAR10-maze')
    parser.add_argument('--N_ini_query', type=int,
                        default=12000)  # 12000 for svhn, 4000 for fashionmnist, mnist; 80000 for cifar10, cifar100; 20000 for svhn
    parser.add_argument('--nc', type=int, default=3)  # 3
    parser.add_argument('--img_size', type=int, default=32)  # 32
    parser.add_argument('--attack_type', type=str, default='DFME', choices=['DFME', 'MAZE', 'DISGUIDE'])
    parser.add_argument('--nz', type=int, default=100, help="Size of random noise input to generator (256)")
    main_args = parser.parse_args()

    device = torch.device('cuda')
    misinformation_model, _ = Blackbox.from_modeldir_split(cfg.misinformation_dir, device)
    misinformation_model.get_model().eval()

    cfg.batch_size = 192  #32 for FMNIST, 192 for CIFAR10 256 for CIFAR10
    #
    # cfg.attack_set = 'cifar_dfme_base'
    # cfg.attack_model_arch = 'resnet24'#'over_resnet18'#'lenet_tl'
    # main_args.ini = False#False  # 2
    # cfg.over_factor = 1  # 3
    # main_runner(main_args)

    # cfg.attack_set = 'cifar_dfme_ini_base'
    # cfg.attack_model_arch = 'resnet24'#'over_resnet18'#'lenet_tl'
    # main_args.ini = True#False  # 2
    # cfg.over_factor = 1  # 3
    # main_runner(main_args)
    #
    # cfg.attack_set = 'cifar-dfme_over' #1
    # cfg.attack_model_arch = 'over_resnet24'
    # # cfg.attack_model_arch = 'over_lenet5'
    # main_args.ini = False  # 2
    # cfg.over_factor = 5  # 3
    # main_runner(main_args)
    #
    # cfg.attack_set = 'cifar-dfme_ini'  # 1
    # cfg.attack_model_arch = 'over_resnet24'
    # # cfg.attack_model_arch = 'over_lenet5'
    # main_args.ini = True  # 2
    # cfg.over_factor = 5  # 3
    # main_runner(main_args)

    # cfg.attack_set = 'svhn-dfme_base'
    # cfg.attack_model_arch = 'resnet18'  # over_vgg16' #64
    # main_args.ini = False  # 2
    # main_args.attack_type = 'DFME'
    # cfg.over_factor = 1  # 3
    # main_runner(main_args)

    # cfg.attack_set = 'svhn-dfme_base-ini'
    # cfg.attack_model_arch = 'over_resnet18' #over_vgg16' #64 24
    # main_args.ini = True  # 2
    # main_args.attack_type = 'DFME'
    # cfg.over_factor = 1  # 3
    # main_runner(main_args)

    # cfg.attack_set = 'cifar-dfme_over'
    # cfg.attack_model_arch = 'over_resnet18'  # over_vgg16' #64
    # main_args.ini = False  # 2
    # main_args.attack_type = 'DFME'
    # cfg.over_factor = 3  # 3
    # main_runner(main_args)

    cfg.attack_set = 'cifar-dfme_ini'
    cfg.attack_model_arch = 'over_resnet18'  # over_vgg16' #64
    main_args.ini = True  # 2
    main_args.attack_type = 'DFME'
    cfg.over_factor = 3  # 3
    main_runner(main_args)

    # cfg.batch_size = 128#256
    # cfg.attack_set = 'fmnist-disguide_over'
    # # # cfg.attack_model_arch = 'over_natnet'
    # cfg.attack_model_arch = 'over_lenet5'
    # # cfg.attack_model_arch = 'over_resnet18'
    # main_args.ini = False
    # cfg.over_factor = 5
    # main_args.attack_type = 'DISGUIDE'
    # main_runner(main_args)

    # cfg.attack_set = 'fmnist-disguide_base'
    # # # cfg.attack_model_arch = 'over_natnet'
    # cfg.attack_model_arch = 'lenet_tl'
    # # cfg.attack_model_arch = 'over_resnet18'
    # main_args.ini = False
    # cfg.over_factor = 1
    # main_args.attack_type = 'DISGUIDE'
    # main_runner(main_args)

    # cfg.attack_set = 'fmnist-disguide_base_ini'
    # # # cfg.attack_model_arch = 'over_natnet'
    # cfg.attack_model_arch = 'over_lenet5'
    # # cfg.attack_model_arch = 'over_resnet18'
    # main_args.ini = True
    # cfg.over_factor = 1
    # main_args.attack_type = 'DISGUIDE'
    # main_runner(main_args)

    # cfg.attack_set = 'fmnist-maze_over'
    # # # cfg.attack_model_arch = 'over_natnet'
    # cfg.attack_model_arch = 'over_lenet5'
    # # cfg.attack_model_arch = 'over_resnet18'
    # main_args.ini = False
    # cfg.over_factor = 5
    # main_runner(main_args)

    # cfg.attack_set = 'fmnist-disguide_ini'
    # # # cfg.attack_model_arch = 'over_natnet'
    # cfg.attack_model_arch = 'over_lenet5'
    # # cfg.attack_model_arch = 'over_resnet18'
    # main_args.ini = True
    # cfg.over_factor = 5
    # main_args.attack_type = 'DISGUIDE'
    # main_runner(main_args)

    # cfg.attack_set = 'fmnist-maze_over,mis'
    # cfg.attack_model_arch = 'over_lenet5'#'over_resnet18'
    # main_args.ini = False  # True
    # cfg.over_factor = 5
    # main_runner(main_args)

    # cfg.attack_set = 'fmnist-maze_base,mis'
    # cfg.attack_model_arch = 'lenet_tl'  # 'resnet18'
    # main_args.ini = False  # True
    # cfg.over_factor = 1
    # main_runner(main_args)

    # cfg.attack_set = 'fmnist_dfme_base,mis'
    # cfg.attack_model_arch = 'lenet_tl'  # 'resnet18'  # 'lenet_tl'
    # main_args.ini = False  # 2
    # cfg.over_factor = 1  # 3
    # main_runner(main_args)

    # cfg.attack_set = 'cifar-maze_ini_base'
    # cfg.attack_model_arch = 'resnet24'
    # main_args.ini = True
    # cfg.over_factor = 1  # 3 this over_factor can not be used. so the base is still over.
    # main_runner(main_args)
    #
    # cfg.attack_set = 'cifar-maze_base'
    # cfg.attack_model_arch = 'resnet24'
    # main_args.ini = False
    # cfg.over_factor = 1  # 3 this over_factor can not be used. so the base is still over.
    # main_runner(main_args)

    # cfg.attack_set = 'cifar-maze_over'
    # cfg.attack_model_arch = 'over_resnet24'
    # main_args.ini = False
    # cfg.over_factor = 5  # 3 this over_factor can not be used. so the base is still over.
    # main_runner(main_args)

    # cfg.attack_set = 'svhn-maze_ini'
    # cfg.attack_model_arch = 'over_resnet24'
    # main_args.ini = True
    # cfg.over_factor = 5  # 3 this over_factor can not be used. so the base is still over.
    # main_runner(main_args)

    # cfg.attack_set = 'svhn-dfme_base'
    # # # cfg.attack_model_arch = 'natnet'
    # cfg.attack_model_arch = 'over_resnet50'
    # main_args.ini = False
    # cfg.over_factor = 1 #3
    # main_runner(main_args)
    # print(maze_base)

    # cfg.attack_set = 'svhn-dfme_base-ini'
    # # # cfg.attack_model_arch = 'natnet'
    # cfg.attack_model_arch = 'over_resnet50'
    # main_args.ini = True
    # cfg.over_factor = 1  # 3
    # main_runner(main_args)

    # cfg.attack_set = 'svhn-maze_ini'
    # # # cfg.attack_model_arch = 'natnet'
    # cfg.attack_model_arch = 'over_resnet50'
    # main_args.ini = True
    # cfg.over_factor = 1  # 3
    # main_runner(main_args)
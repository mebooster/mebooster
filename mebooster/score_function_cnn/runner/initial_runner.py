import argparse
import os

import torch
import yaml
from sklearn.linear_model import ridge_regression

from torch import autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset

import datasets
import zoo
from blackbox import Blackbox
import config as cfg
from dsm_loss import anneal_dsm_score_estimation
from dsm_main import dict2namespace
from gaussian_train import soft_cross_entropy, TransferSetGaussian
from gmm import GaussianMixture
import numpy as np

from ncsnv2 import NCSNv2Simple, get_sigmas, NCSNv2SimpleS2
from no_tenfact import no_tenfact
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from sympy import *
from math import pi

from datetime import datetime

#x
# from vae.variational_autoencoder import cycle, Model, add_args, VariationalFlow, VariationalMeanField, evaluate

import time

from sec_dsm_main import batch_ger, batch_ger3


def tensor_decomposition(T, k):
    N = 2
    R = 2  # number of initialization
    sigma = 0.01
    v = torch.zeros([R, N, k])
    for tol in range(0, R):
        # intialize v0 with SVD-based mothod.
        v[tol, 0, :] = svd_initialization(T, sigma, k)
        for t in range(1, N):
            # comput power iteration update
            v[tol, t, :] = power_iteration_update(T, v[tol, t - 1, :], k)
            # v[t][tol]=T(I, v[t-1][tol], v[t-1][tol])/torch.norm(T(I, v[t-1][tol], v[t-1][tol])) #(31)
            # T(I, u, u) = sum( sum( uj*ul*T[:,j,l]\in R^d

    Tvvv = torch.zeros([R])
    for tol1 in range(0, R):
        Tvvv[tol1] = t_v_v_v(T, v[tol, N - 1, :], k)
    tol_max = torch.argmax(Tvvv)  # tol \in [R]
    # power iteration
    v2 = torch.zeros([N, k])
    v2[0, :] = v[tol_max, N, :]
    for t2 in range(1, N):
        v2[t2, :] = power_iteration_update(T, v2[t2 - 1, :], k)
    # T(I, v2[t2-1], v2[t2-2])/torch.norm(T(I, v2[t2-1], v2[t2-2]))
    v_hat = v2[N - 1, :]  # eigenvector
    mu_hat = t_v_v_v(T, v2[N - 1, :], k)  # mu eigenvalue, scalar, v is eigenvector, vector
    deflated_T = T - mu_hat * ger3(v_hat, v_hat, v_hat)

    return v_hat, mu_hat, deflated_T

def get_teacher_model(device):
    blackbox_dir = cfg.VICTIM_DIR
    blackbox, num_classes = Blackbox.from_modeldir_split(blackbox_dir, device)
    blackbox.eval()
    return blackbox

def convert(*cfg):
    return tuple([v.float().cuda() for v in cfg])
#x
def svd_initialization(T_n, sigma, k):
    theta = torch.zeros([int(torch.log(1 / sigma))])
    u1 = torch.zeros([int(torch.log(1 / sigma)), k, k])
    u1t = torch.zeros([int(torch.log(1 / sigma)), k])
    for tol0 in range(int(torch.log(1 / sigma))):
        theta[tol0] = torch.randn([k])  # nodes
        for l in range(k):
            u1[tol0] = u1[tol0] + theta[tol0][l] * T_n[:, :, l]  # T(I, I, theta)
        u1[tol0] = u1[tol0] / k
        u1t[tol0] = u1[tol0][:, 0]  # top left
        min_val = torch.min(u1t, dim=1).values
        v0 = u1t[torch.max(min_val, dim=0).indices, :]  # the variable is tol0
    return v0

#x
def power_iteration_update(T_n, v_p, k):
    tiuu = torch.zeros([k])
    for j in range(k):
        for l in range(k):
           tiuu = tiuu + T_n[:, j, l]*v_p[j]*v_p[l]
    vt = tiuu/torch.norm(tiuu)
    return vt

def t_v_v_v(T_n, v, k):
    T = 0
    for  i in range(k):
        for j in range(k):
            for l in range(k):
                T = T + T_n[i, j, l]*v[i]*v[j]*v[l]
    return T

def t_w_w_w(T, W, d, k, device):
    T_n = torch.zeros([k, k, k]).to(device)
    for i in range(d):
        for j in range(d):
            for l in range(d):
                T_n = T_n + T[i, j, l] * torch.ger(torch.ger(W[i, :], W[j, :]).view(-1), W[l, :]).reshape(
                    [k, k, k])
    return T_n

def t_i_i_v(T, alpha, d, device):
    M2 = torch.zeros([d, d]).to(device)
    for i in range(d):
        M2 = M2 + T[:, :, i]*alpha[i]
    return M2

def ger3(v_hat, v_hat1, v_hat2):
    M1 = torch.ger(v_hat, v_hat1).view(len(v_hat)*len(v_hat1))
    M = torch.ger(M1, v_hat2).view(len(v_hat), len(v_hat1), len(v_hat2))
    return M

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

def corrMat2corrIdx(score):
    ''' Given score[N, #candidate],
        for each sample sort the score (in descending order)
        and output corr_table[N, #dict(s_idx, score)]
    '''
    sorted_score, sorted_indices = score.sort(1, descending=True)
    N = sorted_score.size(0)
    n_candidate = sorted_score.size(1)
    # Check the correpsonding weights.
    # print("For each teacher node, sorted corr over all student nodes at layer = %d" % i)
    corr_table = []
    for k in range(N):
        tt = []
        for j in range(n_candidate):
            # Compare the upward weights.
            s_idx = int(sorted_indices[k][j])
            score = float(sorted_score[k][j])
            tt.append(dict(s_idx=s_idx, score=score))
        corr_table.append(tt)
    return corr_table

def eval_direction_cnn(W1, blackbox, device):
    # all_one = torch.zeros(X.size(0), 1, dtype=X.dtype).to(X.device).fill_(1.0)
    # X = torch.cat([X, all_one], dim=1)
    w_rand = torch.randn(W1.shape).to(device)

    num_layer = 1
    for layer in range(num_layer):
        out_channel, in_channel, weight, width = blackbox.conv1.weight.shape #3, 1, 3, 3
        print("blackbox.conv1.weight,", blackbox.conv1.weight.shape)
        b_w = blackbox.conv1.weight.view(out_channel, in_channel*weight*width).T
        # b_b = blackbox.fc1.bias
        s_w = W1 #d x m1
        b_w = (b_w/(torch.norm(b_w, dim=0).view(1, -1))).to(device)
        s_w = (s_w/(torch.norm(s_w, dim=0).view(1, -1))).to(device) #d * m1

        w_rand = (w_rand / (torch.norm(w_rand, dim=0).view(1, -1))).to(device)
        recover_error = 0.0
        random_error = 0.0
        mi_ra_error = 0.0
        for dim in range(s_w.shape[1]): #m1
            mi_err1 = torch.sqrt(torch.sum(torch.square(b_w[:, dim].view(-1, 1) - s_w), dim=0))
            mi_err2 = torch.sqrt(torch.sum(torch.square(b_w[:, dim].view(-1, 1) + s_w), dim=0))

            ra_err1 = torch.sqrt(torch.sum(torch.square(b_w[:, dim].view(-1, 1) - w_rand), dim=0))
            ra_err2 = torch.sqrt(torch.sum(torch.square(b_w[:, dim].view(-1, 1) + w_rand), dim=0))
            ra_err = torch.vstack((ra_err1, ra_err2))
            mi_err = torch.vstack((mi_err1, mi_err2))

            mi_ra_err1 = torch.sqrt(torch.sum(torch.square(s_w - w_rand), dim=0))
            mi_ra_err2 = torch.sqrt(torch.sum(torch.square(s_w + w_rand), dim=0))
            mi_ra_err = torch.vstack((mi_ra_err1, mi_ra_err2))

            recover_error = recover_error + torch.min(torch.min(mi_err, dim=0).values).cpu().detach().numpy()
            random_error = random_error + torch.min(torch.min(ra_err, dim=0).values).cpu().detach().numpy()
            mi_ra_error = mi_ra_error + torch.min(torch.min(mi_ra_err, dim=0).values).cpu().detach().numpy()

        print("recovery error:", recover_error / s_w.shape[1])
        print("random_error:", random_error / s_w.shape[1])
        print("mi_ra_error:", mi_ra_error / s_w.shape[1])
    pass

def eval_hid_direction(W1, blackbox,device):
    # all_one = torch.zeros(X.size(0), 1, dtype=X.dtype).to(X.device).fill_(1.0)
    # X = torch.cat([X, all_one], dim=1)
    w_rand = torch.randn(W1.shape).to(device)

    num_layer = 1
    for layer in range(num_layer):
        b_w = blackbox.fc1.weight.T #m1 x d
        # b_b = blackbox.fc1.bias
        s_w = W1 #d x m1
        b_w = (b_w/(torch.norm(b_w, dim=0).view(1, -1))).to(device)
        s_w = (s_w/(torch.norm(s_w, dim=0).view(1, -1))).to(device)

        w_rand = (w_rand / (torch.norm(w_rand, dim=0).view(1, -1))).to(device)
        recover_error = 0.0
        random_error = 0.0
        for dim in range(s_w.shape[1]):
            mi_err1 = torch.sqrt(torch.sum(torch.square(b_w[:, dim].view(-1, 1) - s_w), dim=0))
            mi_err2 = torch.sqrt(torch.sum(torch.square(b_w[:, dim].view(-1, 1) + s_w), dim=0))

            ra_err1 = torch.sqrt(torch.sum(torch.square(b_w[:, dim].view(-1, 1) - w_rand), dim=0))
            ra_err2 = torch.sqrt(torch.sum(torch.square(b_w[:, dim].view(-1, 1) + w_rand), dim=0))
            ra_err = torch.vstack((ra_err1, ra_err2))
            mi_err = torch.vstack((mi_err1, mi_err2))
            recover_error = recover_error + torch.min(torch.min(mi_err, dim=0).values).cpu().detach().numpy()
            random_error = random_error + torch.min(torch.min(ra_err, dim=0).values).cpu().detach().numpy()

        print("recovery error:", recover_error / s_w.shape[1])
        print("random_error:", random_error / s_w.shape[1])
    pass

def power_method(T, M2, d, m1, total, device):
    C = 3 * torch.norm(M2)
    V1 = torch.randn([d, m1]).to(device)
    V2 = torch.randn([d, m1]).to(device)
    # V1 = torch.qr(V1)
    # V2 = torch.qr(V2)

    M2_1 = (C * torch.eye(M2.shape[0]).to(device) + M2).float()
    M2_2 = (C * torch.eye(M2.shape[0]).to(device) - M2).float()
    print("M2_1", M2_1.shape)
    total = total
    for t in range(total):
        V1 = torch.qr(torch.matmul(M2_1, V1)).Q
        V2 = torch.qr(torch.matmul(M2_2, V2)).Q

    lam1 = torch.zeros([m1])
    lam2 = torch.zeros([m1])
    for i in range(m1):
        lam = torch.matmul(V1[:, i].T, M2)
        lam1[i] = torch.abs(torch.matmul(lam, V1[:, i]))

        lam = torch.matmul(V2[:, i].T, M2)
        lam2[i] = torch.abs(torch.matmul(lam, V2[:, i]))

    # print("lam1", lam1)
    # print("lam2", lam2)
    lam3 = torch.cat((lam1, lam2))
    top_eigen_indx = torch.topk(lam3, m1).indices #[10, 3, 4, 1] m1=5
    # print("top_eigen_indx,", top_eigen_indx)
    pi1 = [] #index
    pi2 = [] #index
    for ie in range(len(top_eigen_indx)):
        if top_eigen_indx[ie]<m1:
            pi1.append(int(top_eigen_indx[ie]))
        else:
            pi2.append(int(top_eigen_indx[ie]-m1))
    # k1 = len(pi1) #length
    # k2 = len(pi2) #length

    # print("pi1,", pi1)
    # print("pi2", pi2)
    V1_pr = V1[:, pi1]
    V2_pr = V2[:, pi2]
    print("V1_pr", V1_pr.shape)
    print("V2_pr", V2_pr.shape)
    if V2_pr.shape[1] > 0:
        V2_pr = torch.qr(torch.matmul((torch.eye(d).to(device) - torch.matmul(V1_pr, V1_pr.T)), V2_pr)).Q
        V = torch.cat((V1_pr, V2_pr), dim=1) #d x m1
    else:
        V = V1_pr
    R3 = t_w_w_w(T, V, d, m1, device)
    print("V", V.shape) #d m1
    print("R3", R3.shape) #m1, m1, m1 (pxpxp)
    return V, R3

def KCL(T, p, k, device):
    nojd_err =0.0
    nojd_err1 = 0.1
    sweeps_nojd = [0., 0.]
    #non orthognal joint diagonalization
    print("T", T.shape)
    V1, _, misc = no_tenfact(T, L=2*p, k=k, device=device) #misc is a set
    # V_nojd = misc["V0"] #???
    # sweeps = misc["sweeps"]
    # sweeps_nojd = sweeps_nojd + sweeps
    # sum_err = 0.0
    # print("V0,", misc['V0'].shape)
    return misc['V0'].to(device)  # p x k

def Parser():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--victim_model_dir', metavar='PATH', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"',
                        default=None)
    return parser

def tilde_ger(x, device):
    d = x.shape[0]
    til_m3 = torch.zeros([d, d, d]).to(device)
    I = torch.eye(d).to(device)
    for i in range(d):
        til_m3 = til_m3 + ger3(x, I[i, :], I[i, :]) + ger3(I[i, :], x, I[i, :]) + ger3(I[i, :], I[i, :], x)
        #torch.ger(torch.ger(x, I[i, :]).view(-1), I[i, :]).view(d, d, d) \
        #       + torch.ger(torch.ger(I[i, :], x).view(-1), I[i, :]).view(d, d, d) + \
        #       torch.ger(torch.ger(I[i, :], I[i, :]).view(-1), x).view(d, d, d)
    return til_m3

def calculate_gaussian_score_functions(X_train, y, N_query, d, device, alpha=None):
    # compute T
    # S3 = torch.zeros([N_query, d, d, d]).to(device)
    # S2 = torch.zeros([N_query, d, d]).to(device)
    #
    T = torch.zeros([d, d, d]).to(device)
    # T = torch.zeros([d, d]).to(device)
    M2 = torch.zeros([d, d]).to(device)
    if alpha is None:
        alpha = torch.rand(d)

    # for i in range(int(N_query/2)):
    #     if i % 100 == 0:
    #         print("number,", i)
    #     x = X_train[i, :]
    #     # S2 = torch.ger(x, x) - torch.eye(d).to(device)  # -torch.ger(S2, log_gradient)
    #     # S2 = torch.ger(torch.ger(x, x).view(-1), x).view(d, d, d) - tilde_ger(x, device)
    #     S2 = ger3(x, x, x) - tilde_ger(x, device)
    #     M2 = M2 + y[i] * S2
    for i in range(int(N_query)):
        if i % 1000 == 0:
            print("number,", i)
        x = X_train[i, :]
        S3 = ger3(x, x, x) - tilde_ger(x, device)
        T = T + y[i] * S3
        S2 = torch.ger(x, x) - torch.eye(d).to(device)
        M2 = M2 + y[i] * S2
        # print("M2", M2.shape)
        # print("T", T)
    # for i in range(int(N_query)):
    #     if i % 100 == 0:
    #         print("number,", i)
    #     x = X_train2[i, :]
    #     # S3 = ger3(x, x, x) - tilde_ger(x, device)
    #     # T = T + y2[i] * S3
    #     S2 = torch.ger(x, x) - torch.eye(d).to(device)
    #     M2 = M2 + y2[i] * S2
    #     # print("M2", M2.shape)
    #     # print("T", T)
    # compute T
    T = T / N_query  # (d,d,d)
    # M2 = M2 / N_query  # (d d)
    M2 = t_i_i_v(T, alpha, d, device)
    return M2, T, alpha  #P2, P3

def t_v_i_i_i(T, alpha, device):
    d = T.shape[1]
    T_1 = torch.zeros(d, d, d).to(device)
    for i in range(len(alpha)):
        T_1 += alpha[i] * T[i, :, :, :]
    return T_1

def t_v_i_i(M2, alpha, device):
    d = M2.shape[1]
    M2_1 = torch.zeros(d, d).to(device)
    for i in range(len(alpha)):
        M2_1 += alpha[i] * M2[i, :, :]
    return M2_1

def calculate_score_functions(X_train, y, gmm_model, N_query, d, num_classes, device):
    # compute T
    T = torch.zeros([num_classes, d, d, d]).to(device)
    M2 = torch.zeros([num_classes, d, d]).to(device)
    for i in range(N_query):
        if i % 100 == 0:
            print("number,", i)
        x = X_train[i]
        px = torch.exp(gmm_model._estimate_log_prob(x) + torch.log(gmm_model.pi)).squeeze().sum()
        # print("px,", px)
        gradient = autograd.grad(px, x, create_graph=True)[0]  # the first one is tup
        # print("px", px)
        # print("gradient,", gradient)
        s1_gradient = torch.zeros([d, d]).to(device)
        for ed_d in range(len(gradient)):
            grad_temp = autograd.grad(gradient[ed_d], x, create_graph=True)
            s1_gradient[ed_d, :] = grad_temp[0]
            torch.cuda.empty_cache()

        s2_gradient = torch.zeros([d, d, d]).to(device)
        for ed_d in range(d):
            for rd_d in range(d):
                grad_temp2 = autograd.grad(s1_gradient[ed_d, rd_d], x, create_graph=True)
                with torch.no_grad():
                    s2_gradient[ed_d, rd_d, :] = grad_temp2[0]

        with torch.no_grad():
            s2_gradient = s2_gradient.to(device)
            s1_gradient = s1_gradient.to(device)

            S3 = (-1) * s2_gradient / px  # -torch.ger(S2, log_gradient)
            T = T + torch.ger(y[i, :], S3.view(-1)).view(num_classes, d, d, d)

            S2 = s1_gradient / px
            M2 = M2 + torch.ger(y[i, :], S2.view(-1)).view(num_classes, d, d)
    # compute T
    T = T / N_query
    # M2 = t_i_i_v(T, alpha, d, device)
    M2 = M2 / N_query
    #alpha
    alpha = torch.rand(num_classes).to(device)
    T_1 = t_v_i_i_i(T, alpha, device)
    M2_1 = t_v_i_i(M2, alpha, device)
    return M2_1, T_1

def calculate_score_functions_sympy(X_train, y, gmm_model, N_query, d, num_classes, device):
    # compute T
    T = torch.zeros([num_classes, d, d, d]).to(device)
    M2 = torch.zeros([num_classes, d, d]).to(device)
    x_s = symbols('x0:{}'.format(d))#,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19')
    print("x_s", x_s)
    x_m = Matrix(x_s)
    g_pi = gmm_model.pi[0].cpu().detach().numpy()
    print("pi,", g_pi.shape)
    # var = gmm_model.var.cpu().detach().numpy()
    mu = gmm_model.mu[0].cpu().detach().numpy() #[n, d]
    n_component = gmm_model.n_components
    precision = torch.inverse(gmm_model.var)#.cpu().detach().numpy() #[n, d, d]
    # print("precison,", precision.shape)
    # print("mu,", mu.shape)
    # print("var,", var.shape)
    log_2pi = d * np.log(2. * pi)
    log_det = gmm_model._calculate_log_det(precision)
    log_det = log_det.cpu().detach().numpy()
    precision=precision[0].cpu().detach().numpy()

    print("log_det", log_det.shape)
    print("g_pi", g_pi.shape)
    print("Matrix(precision),", Matrix(precision[0]).shape)

    x_mu = (x_m-Matrix(mu[0, :]))
    x_mu_T = x_mu.T
    x_mu_T_precision_x_mu = x_mu_T * Matrix(precision[0]) * x_mu
    indx = -.5 * (log_2pi - log_det[0, 0] + x_mu_T_precision_x_mu[0])
    px = exp(indx + log(g_pi[0, 0]))
    for i in range(1, n_component):
        x_mu = (x_m - Matrix(mu[i, :]))
        x_mu_T = x_mu.T
        x_mu_T_precision_x_mu = x_mu_T * Matrix(precision[i]) * x_mu
        px = px + exp(-.5 * (log_2pi - log_det[i, 0] + x_mu_T_precision_x_mu[0]) + log(g_pi[i, 0]))
    print(datetime.now())
    diff_2 = diff(px, x_m, 2).reshape(d, d)
    print("diff2.shape", diff_2.shape)
    print(datetime.now())
    diff_3 = diff(diff_2, x_m, 1).reshape(d, d, d)
    print("diff3.shape", diff_3.shape)
    print(datetime.now())
    # print("px,", px)
    s2_gradient = []
    s3_gradient = []
    for j in range(d):
        s2_gradient.append(lambdify((x_s), diff_2[j, :], "numpy"))
        s3_gradient.append(lambdify((x_s), diff_3[j, :, :], "numpy"))
        # for k in range(d):
        #     s3_gradient.append(lambdify((x_s), diff_3[j, k, :], "numpy"))
        print("j", j)
        print(datetime.now())
    px_func = lambdify((x_s), px, "numpy")
    print("start train query")
    for i in range(N_query):
        if i % 100 == 0:
            print("number,", i)
            print(datetime.now())
        x = X_train[i].cpu().detach().numpy().tolist()
        # print("x,", x.shape)
        s2_g = torch.zeros([d, d]).to(device)
        s3_g = torch.zeros([d, d, d]).to(device)

        for j in range(d):
          s2_g[j, :] = torch.tensor(s2_gradient[j](*x)).to(device)
          s3_g[j, :, :] = torch.tensor(s3_gradient[j](*x)).to(device)
          # for k in range(d):
          #     s3_g[j, k, :] = torch.tensor(s3_gradient[int(j*d + k)](*x)).to(device)
        px_0 = torch.tensor(px_func(*x)).to(device)

        with torch.no_grad():
            # s2_gradient = s2_gradient.to(device)
            # s1_gradient = s1_gradient.to(device)

            S3 = (-1) * s3_g / px_0  # -torch.ger(S2, log_gradient)
            T = T + torch.ger(y[i, :], S3.view(-1)).view(num_classes, d, d, d)

            S2 = s2_g / px_0
            M2 = M2 + torch.ger(y[i, :], S2.view(-1)).view(num_classes, d, d)

    # compute T
    T = T / N_query
    # M2 = t_i_i_v(T, alpha, d, device)
    M2 = M2 / N_query

    #alpha
    alpha = torch.rand(num_classes).to(device)
    T_1 = t_v_i_i_i(T, alpha, device)
    M2_1 = t_v_i_i(M2, alpha, device)
    return M2_1, T_1

# def calculate_score_functions_vae(X_train, y, vae_model, variational, N_query, d, num_classes, device, vae_cfg):
#     # compute T
#     T = torch.zeros([num_classes, d, d, d]).to(device)
#     M2 = torch.zeros([num_classes, d, d]).to(device)
#     for i in range(N_query):
#         if i % 100 == 0:
#             print("number,", i)
#         x = X_train[i].unsqueeze(dim=0).to(device)
#
#         z, log_q_z = variational(x, vae_cfg.n_samples)
#         log_p_x_and_z = vae_model(z, x)
#         # importance sampling of approximate marginal likelihood with q(z)
#         # as the proposal, and logsumexp in the sample dimension
#         elbo = log_p_x_and_z - log_q_z
#         px = torch.exp(torch.logsumexp(elbo, dim=1) - np.log(vae_cfg.n_samples))# torch.exp(log_p_x)
#
#         # print("px,", px)
#         gradient = autograd.grad(px, x, create_graph=True)[0]  # the first one is tup
#         # print("px", px)
#         # print("gradient,", gradient)
#         s1_gradient = torch.zeros([d, d]).to(device)
#         for ed_d in range(len(gradient)):
#             grad_temp = autograd.grad(gradient[ed_d], x, create_graph=True)
#             s1_gradient[ed_d, :] = grad_temp[0]
#             torch.cuda.empty_cache()
#
#         s2_gradient = torch.zeros([d, d, d]).to(device)
#         for ed_d in range(d):
#             for rd_d in range(d):
#                 grad_temp2 = autograd.grad(s1_gradient[ed_d, rd_d], x, create_graph=True)
#                 with torch.no_grad():
#                     s2_gradient[ed_d, rd_d, :] = grad_temp2[0]
#
#         with torch.no_grad():
#             s2_gradient = s2_gradient.to(device)
#             s1_gradient = s1_gradient.to(device)
#
#             S3 = (-1) * s2_gradient / px  # -torch.ger(S2, log_gradient)
#             T = T + torch.ger(y[i, :], S3.view(-1)).view(num_classes, d, d, d)
#
#             S2 = s1_gradient / px
#             M2 = M2 + torch.ger(y[i, :], S2.view(-1)).view(num_classes, d, d)
#     # compute T
#     T = T / N_query
#     # M2 = t_i_i_v(T, alpha, d, device)
#     M2 = M2 / N_query
#     # alpha
#     alpha = torch.rand(num_classes).to(device)
#     T_1 = t_v_i_i_i(T, alpha, device)
#     M2_1 = t_v_i_i(M2, alpha, device)
#     return M2_1, T_1

def fit_gaussian(X_train, n_com, d, device):
    # GMM
    gmm_model = GaussianMixture(n_components=n_com, n_features=d)
    gmm_model = gmm_model.to(device)
    gmm_model.fit(X_train, delta=1e-64, n_iter=500)
    # print("gmm_model.mu", gmm_model.mu)#1, 1, 30
    # print("gmm_model.var", gmm_model.var)#1, 1, 30, 30
    return gmm_model

def recoverDir(V, U, d, m1):
    #V dxm1
    #U m1xd
    #W1_dir = m1 x d
    W1_dir = torch.zeros([d, m1])
    W1_dir2 = torch.zeros([d, m1])
    print("V.shape", V.shape)
    for i in range(m1):
        W1_dir[:, i] = torch.matmul(V, U[:, i]) #d x m1, m*1
        # W1_dir2[:, i] = torch.matmul(V, U[i, :])  # d x m1, m*1
    # print("W1_dir", W1_dir[-1, :])
    return W1_dir #d x m1

def ktensor(lambd, a, p=3, k=4, device='cuda'):
    T = torch.zeros([p, p, p]).to(device)
    for i in range(k):
        T = T + lambd[i] * torch.ger(torch.ger(a[:, i], a[:, i]).view(-1), a[:, i]).view(p, p, p)
    return T

def eval_kcl(U, V_true, device):
    w_rand = torch.randn(U.shape).to(device)
    # b_w = (U / (torch.norm(U, dim=1).view(-1, 1))).to(device)
    b_w = V_true
    s_w = U
    # s_w = (V_true / (torch.norm(V_true, dim=1).view(-1, 1))).to(device)
    w_rand = (w_rand / (torch.norm(w_rand, dim=0).view(-1, 1))).to(device)
    b_w = (b_w / (torch.norm(b_w, dim=0).view(-1, 1))).to(device)
    s_w = (s_w / (torch.norm(s_w, dim=0).view(-1, 1))).to(device)
    recover_error = 0.0
    random_error = 0.0
    for dim in range(s_w.shape[1]):
        mi_err1 = torch.sqrt(torch.sum(torch.square(b_w[:, dim].view(-1, 1) - s_w), dim=0))
        mi_err2 = torch.sqrt(torch.sum(torch.square(b_w[:, dim].view(-1, 1) + s_w), dim=0))

        ra_err1 = torch.sqrt(torch.sum(torch.square(b_w[:, dim].view(-1, 1) - w_rand), dim=0))
        ra_err2 = torch.sqrt(torch.sum(torch.square(b_w[:, dim].view(-1, 1) + w_rand), dim=0))
        ra_err = torch.vstack((ra_err1, ra_err2))
        mi_err = torch.vstack((mi_err1, mi_err2))
        recover_error = recover_error + torch.min(torch.min(mi_err, dim=0).values).cpu().detach().numpy()
        random_error = random_error + torch.min(torch.min(ra_err, dim=0).values).cpu().detach().numpy()

    print("recovery error:", recover_error/s_w.shape[1])
    print("random_error:", random_error/s_w.shape[1])
    pass

def calculate_true(V, teacher_model,device):
    out_channel, in_channel, weight, width = teacher_model.conv1.weight.shape
    W1 = teacher_model.conv1.weight.view(out_channel, in_channel*weight*width).T #d * m
    V_true = torch.zeros([W1.shape[1], W1.shape[1]]).to(device)
    for i in range(W1.shape[1]):
        w1 = W1[:, i]/torch.norm(W1[:, i])
        V_true[:, i] = torch.matmul(V, w1)
    return V_true

def eval_model(X_train, teacher_model, student_model, random_model):
    y = teacher_model(X_train)
    y2 = student_model(X_train)
    y3 = random_model(X_train)
    print("recover_model,", torch.sum(torch.abs(y-y2))/y.shape[0])
    print("random_model,", torch.sum(torch.abs(y - y3))/y.shape[0])
    pass

def generate_mm_gaussian(N_query, d, teacher_model, device):
    # X_train = torch.zeros([N_query, d]).to(device)
    mu=np.zeros([10, d])
    mu[1, :] = 1
    mu[2, :] = 2
    mu[3, :] = 3
    mu[4, :] = 4
    mu[5, :] = 5
    mu[6, :] = 6
    mu[7, :] = 7
    mu[8, :] = 8
    mu[9, :] = 9
    rho = np.diag(np.ones([10]))
    x_cls0 = np.random.multivariate_normal(mu[0,:], rho, size=int(N_query / d))
    x_cls1 = np.random.multivariate_normal(mu[1, :], rho, size=int(N_query / d))
    x_cls2 = np.random.multivariate_normal(mu[2, :], rho, size=int(N_query / d))
    x_cls3 = np.random.multivariate_normal(mu[3, :], rho, size=int(N_query / d))
    x_cls4 = np.random.multivariate_normal(mu[4, :], rho, size=int(N_query / d))
    x_cls5 = np.random.multivariate_normal(mu[5,:], rho, size=int(N_query / d))
    x_cls6 = np.random.multivariate_normal(mu[6, :], rho, size=int(N_query / d))
    x_cls7 = np.random.multivariate_normal(mu[7, :], rho, size=int(N_query / d))
    x_cls8 = np.random.multivariate_normal(mu[8, :], rho, size=int(N_query / d))
    x_cls9 = np.random.multivariate_normal(mu[9, :], rho, size=int(N_query / d))

    x = np.vstack([x_cls0, x_cls1])
    x = np.vstack([x, x_cls2])
    x = np.vstack([x, x_cls3])
    x = np.vstack([x, x_cls4])
    x = np.vstack([x, x_cls5])
    x = np.vstack([x, x_cls6])
    x = np.vstack([x, x_cls7])
    x = np.vstack([x, x_cls8])
    x = np.vstack([x, x_cls9])

    X_train = torch.tensor(x).float().to(device)
    batch_size = 20
    y = torch.zeros([N_query, d]).to(device)
    for i in range(int(N_query/batch_size)):
        y[i*batch_size:min((i+1)*batch_size, N_query),:] = teacher_model.deep_half(X_train[i*batch_size:min((i+1)*batch_size, N_query),:])
    return X_train, y

def calculate_score_functions_via_model(x, y, scorenet1, scorenet2, device):
    d = x.shape[1]
    batch = 10 #x.shape[0]

    num_classes = y.shape[1]
    T = torch.zeros([num_classes, d, d, d]).to(device)
    M2 = torch.zeros([num_classes, d, d]).to(device)

    for ib in range(int(x.shape[0]/batch)):
        if ib % 100 == 0:
           print("batch_index", ib)
        y_temp = y[ib*batch :(ib+1)*batch]
        x_temp = x[ib*batch :(ib+1)*batch]
        x_temp = Variable(x_temp)
        x_temp.requires_grad = True
        s1_model = scorenet1(x_temp)
        #s2_model = scorenet2(x_temp).view([batch, d, d])
        auto_grad_s1 = torch.zeros([batch, d, d])
        for i in range(d):
            # print("grad,", i)
            auto_grad_s1[:, i, :] = autograd.grad(torch.sum(s1_model, dim=0)[i], x_temp, create_graph=True)[0]
        s2_model = batch_ger(s1_model, s1_model) + auto_grad_s1

        auto_grad_s2 = torch.zeros([batch, d, d, d])
        for i in range(d):
            # print("grad,", i)
            for j in range(d):
                auto_grad_s2[:, i, j, :] = autograd.grad(torch.sum(s2_model, dim=0)[i, j], x_temp, create_graph=True)[0]
        s3_model = batch_ger3(s2_model.to(device), s1_model.to(device)) + auto_grad_s2

        with torch.no_grad():
            for i in range(len(y_temp)):
                T = T + torch.ger(y_temp[i, :], -s3_model[i].view(-1).to(device)).view(num_classes, d, d, d)
                M2 = M2 + torch.ger(y_temp[i, :], s2_model[i].view(-1).to(device)).view(num_classes, d, d)

    # compute T
    T = T / x.shape[0]
    # M2 = t_i_i_v(T, alpha, d, device)
    M2 = M2 / x.shape[0]
    # alpha
    alpha = torch.rand(num_classes).to(device)
    T_1 = t_v_i_i_i(T, alpha, device)
    M2_1 = t_v_i_i(M2, alpha, device)

    return M2_1, T_1

def train_denoise_score(dataload, scorenet, score_opt, epochs, sigmas):
    scorenet.train()
    step=0
    for epoch in range(epochs):
        for data, target in dataload:
            step += 1
            score_opt.zero_grad()
            dsm_loss = anneal_dsm_score_estimation(scorenet, data, sigmas, None)
            dsm_loss.backward()
            score_opt.step()
        if epoch % 100==0:
            print("[{}]dsm_loss".format(epoch), dsm_loss)
    return scorenet

class InitialRunner():
    def __init__(self):
        self.data_out_path = '../data'
        self.model_out_path = '../model'

    def run(self):
        device = torch.device('cuda')
        N_eval = 1000
        N_train = 20000  # used to calcuate the score functions
        # d = 10#20
        # m1 = 3#10 # the number of nodes in the first layer
        num_classes = 2
        # batch_size =32
        # power method
        total = 2000
        torch.set_printoptions(precision=16)

        # #teacher model
        teacher_model = get_teacher_model(device)  # fc1(weight, bias), fc2(weight, bias)
        print("teacher_model", teacher_model)
        # # data
        # X_train = torch.load(cfg.VICTIM_DIR+'\\x_train.pt').to(device)
        # X_train.requires_grad = True
        # y_o = teacher_model(X_train)  # not real train, 10000
        # y = torch.softmax(y_o, dim=1)
        N_query = 20000  # 000

        # data: use cpu not gpu
        # dataset_name = 'MNIST'
        # valid_datasets = datasets.__dict__.keys()
        # print("valid_datasets:", valid_datasets)
        # if dataset_name not in valid_datasets:
        #     raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
        # dataset = datasets.__dict__[dataset_name]

        # modelfamily = datasets.dataset_to_modelfamily[dataset_name]
        # print("modelfamily", modelfamily)

        # train_transform = datasets.modelfamily_to_transforms[modelfamily]['train']
        # test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']
        # trainset = dataset(train=True, transform=train_transform)  # , download=True
        # trainset = Subset(trainset, np.asarray(range(N_query)))
        # test_dataset = dataset(train=False, transform=test_transform)  # , download=True

        # train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, pin_memory=True)
        # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        # X_train = torch.zeros([len(trainset), 10]).to(device)
        # y = torch.zeros([len(trainset), 10]).to(device)
        # for idx, (data, _) in enumerate(train_loader):
        #     # print("idx,", idx)
        #     X_train[idx * batch_size:min((idx+1)*batch_size, len(trainset)), :] = teacher_model.midx(data.to(device))
        #     y[idx * batch_size:min((idx + 1) * batch_size, len(trainset)), :] = teacher_model(data.to(device))
        #     torch.cuda.empty_cache()
        # torch.save(X_train, 'x_train.pt')
        # torch.save(y, 'y_train.pt')
        # out_eval = teacher_model(X_eval)  # 10000, 2

        X_train = torch.load("data\\x.pt").to(device)
        step_1_x = torch.load('data\\step_1_x.pt').to(device)
        step_2_x = torch.load('data\\step_2_x.pt').to(device)
        step_3_x = torch.load('data\\step_3_x.pt').to(device)
        step_4_x = torch.load('data\\step_4_x.pt').to(device)

        d = step_1_x.shape[1] * step_1_x.shape[2] * step_1_x.shape[3]  # d=9
        N_query = step_1_x.shape[0]  # 20000
        step_1_x = step_1_x.reshape(N_query, -1)
        step_2_x = step_2_x.reshape(N_query, -1)
        step_3_x = step_3_x.reshape(N_query, -1)
        step_4_x = step_4_x.reshape(N_query, -1)

        m1 = 3  # output_channel
        y = teacher_model(X_train)  # torch.load('y_train_20k.pt')[:N_query]
        print("X_train.shape", X_train.shape)
        print("y.shape", y.shape)

        with open(os.path.join('configs', 'dsm.yml'), 'r') as f:
            pre_config = yaml.full_load(f)
        config = dict2namespace(pre_config)
        config.data.dim = d

        state_dict = torch.load('./model/scorenet2_step1.tar.pth')
        scorenet2 = NCSNv2SimpleS2(config).to(device)
        scorenet2.load_state_dict(state_dict)

        scorenet1 = NCSNv2Simple(config).to(device)  # DSM_Generator_S1
        state_dict1 = torch.load('./model/scorenet_step1.tar.pth')
        scorenet1.load_state_dict(state_dict1)

        #####
        M2, T = calculate_score_functions_via_model(step_1_x.to(device), y.to(device), scorenet1, scorenet2,
                                                    device)  # X_train
        # print("M2", M2.shape)
        # print("T", T.shape)

        # print("gmm_model")
        # n_com=1
        # gmm_model = fit_gaussian(step_1_x, n_com, d, device)#step1
        print("calculate score functions")
        # M2, T = calculate_score_functions_sympy(step_1_x, y, gmm_model, N_query, d, num_classes, device) #try this

        # # there y is a scalar
        print("calculate gaussian score functions")
        # M2, T, alpha = calculate_gaussian_score_functions(X_train, y, N_query, d, device)

        print("power method")
        V, R3 = power_method(T, M2, d, m1, total, device)  # q=20, m1=20 V=[d, m1]

        p = m1
        k = m1
        print("non-orthognal tensor decomposition")
        # no
        U = KCL(T=R3, p=p, k=k, device=device)  # R3\in m1 x m1 x m1; U=[m1, m1] u=V^T v, [p, k]
        # what's k, p V_true = [p k],
        # eval_kcl(U, V_true, device)
        V_true = calculate_true(V.T, teacher_model, device)
        print("U")
        eval_kcl(U, V_true, device)  # k x k

        # print("recover directions")
        # no
        W1_dir_step1 = recoverDir(V, U, d, m1)  # but we don't know U we need column or others????  use column first
        # torch.save(W1_dir, 'w1_dir.pt')

        # calculate for step 2, 3, 4
        # gmm_model = fit_gaussian(step_2_x, n_com, d, device)  # step1
        # M2, T = calculate_score_functions_sympy(step_2_x, y, gmm_model, N_query, d, num_classes, device)  # try this
        # V, R3 = power_method(T, M2, d, m1, total, device)
        # p = m1
        # k = m1
        # print("non-orthognal tensor decomposition")
        ## no
        # U = KCL(T=R3, p=p, k=k, device=device)
        # W1_dir_step2 = recoverDir(V, U, d, m1)

        # W1_dir = torch.zeros_like(W1_dir_step1).to(device)#d * m1
        # W1_dir[:2, :] = W1_dir_step1[:2, :]
        # W1_dir[2, :] = W1_dir_step2[2, :]
        # W1_dir[3:5, :] = W1_dir_step1[3:5, :]
        # W1_dir[5, :] = W1_dir_step2[5, :]
        # W1_dir[6:8, :] = W1_dir_step3[6:8, :]
        # W1_dir[8, :] = W1_dir_step4[8, :]

        W1_dir = W1_dir_step1
        print("W1_dir")
        print("Compare W1_dir with random")
        eval_direction_cnn(W1_dir, teacher_model, device)

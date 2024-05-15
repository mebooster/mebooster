import argparse
import functools
import os

import torch
import torch.autograd as autograd
import numpy as np
import yaml
from sympy import *
from torch.utils.data import Dataset, DataLoader

from gmm import GaussianMixture
from datetime import datetime
from math import pi
from ssm import Img_MLPScore, Img_Score
from score_utils import *

class TransferSetGaussian(Dataset):
    def __init__(self, x_train, y):
        self.data = x_train
        self.targets = y

    def __getitem__(self, index):
        x, target = self.data[index, :], self.targets[index]

        return x, target

    def __len__(self):
        return len(self.data)

def calculate_score_functions_sympy(X_train, gmm_model, N_query, d, device):
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
    diff_1 = diff(px, x_m, 1)
    # diff_2 = diff(px, x_m, 2).reshape(d, d)
    # print("diff2.shape", diff_2.shape)
    # print(datetime.now())
    # diff_3 = diff(diff_2, x_m, 1).reshape(d, d, d)
    # print("diff3.shape", diff_3.shape)
    # print(datetime.now())
    # print("px,", px)
    # s2_gradient = []
    # s3_gradient = []
    # for j in range(d):
    #     s2_gradient.append(lambdify((x_s), diff_2[j, :], "numpy"))
    #     s3_gradient.append(lambdify((x_s), diff_3[j, :, :], "numpy"))
    #     # for k in range(d):
    #     #     s3_gradient.append(lambdify((x_s), diff_3[j, k, :], "numpy"))
    #     print("j", j)
    #     print(datetime.now())

    s1_gradient = lambdify((x_s), diff_1, "numpy")
    px_func = lambdify((x_s), px, "numpy")
    print("start train query")
    S1 = torch.zeros([N_query, d]).to(device)
    for i in range(N_query):
        if i % 100 == 0:
            print("number,", i)
            print(datetime.now())
        x = X_train[i].cpu().detach().numpy().tolist()
        # print("x,", x.shape)
        # s1_g = torch.zeros([d]).to(device)
        # s2_g = torch.zeros([d, d]).to(device)
        # s3_g = torch.zeros([d, d, d]).to(device)

        s1_g = torch.tensor(s1_gradient(*x)).squeeze().to(device)
        # for j in range(d):
          # s2_g[j, :] = torch.tensor(s2_gradient[j](*x)).to(device)
          # s3_g[j, :, :] = torch.tensor(s3_gradient[j](*x)).to(device)

          # for k in range(d):
          #     s3_g[j, k, :] = torch.tensor(s3_gradient[int(j*d + k)](*x)).to(device)
        px_0 = torch.tensor(px_func(*x)).to(device)

        with torch.no_grad():
            # s2_gradient = s2_gradient.to(device)
            # s1_gradient = s1_gradient.to(device)

            # S3 = (-1) * s3_g / px_0  # -torch.ger(S2, log_gradient)
            # S2 = s2_g / px_0

            # print(s1_g.shape)
            # print(px_0.shape)

            S1[i] = (-1) * s1_g / px_0

    return S1 #S2, S3

def calculate_score_functions(X_train, gmm_model, N_query, d, device):
    # compute T
    S1 = torch.zeros([N_query, d]).to(device)
    for i in range(N_query):
        if i % 100 == 0:
            print("number,", i)
        x = X_train[i]
        px = torch.exp(gmm_model._estimate_log_prob(x) + torch.log(gmm_model.pi)).squeeze().sum()
        # print("px,", px)
        gradient = autograd.grad(px, x, create_graph=True)[0]  # the first one is tup
        # print("px", px)
        # print("gradient,", gradient.shape)
        with torch.no_grad():
            S1[i, :] = (-1) * gradient / px  # -torch.ger(S2, log_gradient)

    return S1


def generate_gmm_data(N_query, d, n_com, device):
    # X_train = torch.zeros([N_query, d]).to(device)
    mu=np.zeros([n_com, d])
    rho = np.diag(np.ones([d]))
    x = np.random.multivariate_normal(mu[0, :], rho, size=int(N_query / n_com))
    for i in range(1, n_com):
        mu[i, :] = i + 3
        clsi = np.random.multivariate_normal(mu[i, :], rho, size=int(N_query / n_com))
        x = np.vstack([x, clsi])
    X_train = torch.tensor(x).float().to(device)

    return X_train, mu, rho

def fit_gaussian(X_train, n_com, d, device):
    # GMM
    gmm_model = GaussianMixture(n_components=n_com, n_features=d)
    gmm_model = gmm_model.to(device)
    gmm_model.fit(X_train, delta=1e-64, n_iter=500)
    # print("gmm_model.mu", gmm_model.mu)#1, 1, 30
    # print("gmm_model.var", gmm_model.var)#1, 1, 30, 30
    return gmm_model


def eval_l2_distance(s1_1, s1_2):
    error = 0.0
    for i in range(len(s1_1)):
        # print(scores1[i] - rand1[i])
        error += torch.sum(torch.pow(s1_1[i] - s1_2[i], 2))
    print("minus_error,", error/len(s1_1))

    for i in range(len(s1_1)):
        # print(scores1[i] - rand1[i])
        error += torch.sum(torch.pow(s1_1[i] + s1_2[i], 2))
    print("plus_error,", error/len(s1_1))
    pass

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def train_score(dataload, scorenet1, score_opt, epochs, n_particles):
    for epoch in range(epochs):
        for data, target in dataload:
            score_opt.zero_grad()
            ssm_loss, *_ = sliced_score_estimation_fir(scorenet1, data, n_particles=n_particles)
            ssm_loss.backward()
            score_opt.step()
        if epoch % 100==0:
            print(ssm_loss)
        if epoch %1000 == 0:
            print(scorenet1(data[0:5]))
    return scorenet1


def main():
    # parse config file
    with open(os.path.join('configs', 'ssm.yml'), 'r') as f:
        pre_config = yaml.full_load(f)
    config = dict2namespace(pre_config)

    #args
    device = 'cuda:0'
    d = 5
    N_query = 20000
    n_com = 5
    batch_size=128

    #data
    print("data_generate")
    x, gt_mu, gt_var = generate_gmm_data(N_query, d, n_com, device) #
    y = torch.randn([N_query, 1]).to(device)
    dataset = TransferSetGaussian(x, y)
    dataload = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # true_gmm_model = GaussianMixture(n_components=n_com, n_features=d).to(device)
    # true_gmm_model.mu.data[0] = torch.tensor(gt_mu).to(device)
    # true_gmm_model.var.data[0][:] = torch.diag(torch.ones([d])).to(device)
    # true_gmm_model.pi.data[0] = (torch.ones([d,1])/d).to(device)
    # print(true_gmm_model.mu.data[0]) #1, 10, 10
    # print(true_gmm_model.var[0]) #1, 10, 10, 10
    # print(true_gmm_model.pi.data[0]) #1, 10, 1

    # true_scores1 = calculate_score_functions_sympy(x, true_gmm_model, N_query,  d, device)
    # x.requires_grad = True
    # true_scores1 = calculate_score_functions(x, true_gmm_model, N_query, d, device)
    #gmm_model
    print("gmm_model")
    gmm_model = fit_gaussian(x, n_com, d, device)
    print("calculate_score")
    scores1 = calculate_score_functions_sympy(x, gmm_model, N_query, d, device) #scores2, scores3

    # print("true_scores vs sympy_scores")
    # eval_l2_distance(true_scores1, scores1)

    scorenet1 = Img_MLPScore(config).to(device)# if config.data.dataset == 'MNIST' else \
                #Img_Score(config).to(device) #104
    s_model1 = scorenet1(x)
    print("init_model_score vs sympy_score")
    eval_l2_distance(s_model1, scores1)
    score_opt = torch.optim.RMSprop(scorenet1.parameters(), lr=config.optim.lr)
    scorenet1 = train_score(dataload, scorenet1, score_opt, config.training.n_epochs, config.model.n_particles)
    # scorenet2 =
    # scorenet3 =
    s_model1 = scorenet1(x)
    # s_model2 = scorenet2(x)
    # s_model3 = scorenet3(x)
    rand1 = torch.randn([N_query, d]).to(device)
    # rand2 = torch.randn([d,d]).to(device)
    # rand3 = torch.randn([d,d,d]).to(device)

    print("eval_l2_distance")
    print("rand_score vs sympy_score")
    eval_l2_distance(scores1, rand1)
    print("trained_model_score vs sympy_score")
    eval_l2_distance(s_model1, scores1)
    print("rand1", rand1[0:5])
    print("s_model1", s_model1[0:5])
    print("scores1", scores1[0:5])

if __name__ == '__main__':
    main()
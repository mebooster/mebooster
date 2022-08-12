import os

import yaml
from torch import autograd

from gmm import GaussianMixture
from main import dict2namespace
import torch

from ssm import Img_MLPScore
import numpy as np


def generate_gmm_data(N_query, d, n_com, device):
    # X_train = torch.zeros([N_query, d]).to(device)
    mu=np.zeros([n_com, d])
    rho = np.diag(np.ones([d]))
    x = np.random.multivariate_normal(mu[0,:], rho, size=int(N_query / n_com))
    for i in range(1, n_com):
        mu[i, :] = i
        clsi = np.random.multivariate_normal(mu[i, :], rho, size=int(N_query / n_com))
        x = np.vstack([x, clsi])
    X_train = torch.tensor(x).float().to(device)

    return X_train, mu, rho

def maintest_():
    # parse config file
    with open(os.path.join('configs', 'ssm.yml'), 'r') as f:
        pre_config = yaml.full_load(f)
    config = dict2namespace(pre_config)

    # args
    device = 'cuda:0'
    d = 10
    N_query = 10000
    n_com = 5
    batch_size = 100
    torch.set_printoptions(precision=16)

    # data
    print("data_generate")
    X_train, gt_mu, gt_var = generate_gmm_data(N_query, d, n_com, device)  #
    X_train.requires_grad = True

    print("X_train.shape", X_train.shape)

    true_gmm_model = GaussianMixture(n_components=n_com, n_features=d).to(device)
    true_gmm_model.mu.data[0] = torch.tensor(gt_mu).to(device)
    true_gmm_model.var.data[0][:] = torch.diag(torch.ones([d])).to(device)
    true_gmm_model.pi.data[0] = (torch.ones([n_com, 1])/n_com).to(device)

    scorenet1 = Img_MLPScore(config).to(device)

    sum1 = torch.zeros([d]).to(device)
    sum2 = torch.zeros([d]).to(device)
    # sum3 = torch.zeros([d]).to(device)

    for i in range(int(N_query)):
        if i % 100 == 0:
            print("number,", i)
        # x = X_train[i*batch_size : (i+1)*batch_size]
        x = X_train[i]
        px = torch.sum(torch.exp(true_gmm_model._estimate_log_prob(x) + torch.log(true_gmm_model.pi)))
        # print("px,", px.shape)
        gradient = autograd.grad(px, x, create_graph=True)[0]  # the first one is tup
        fx_grad = []
        for i in range(d):
            fx_grad.append(autograd.grad(scorenet1(x)[i], x, create_graph=True)[0][i])

        # print("px", px)
        # print("gradient,", gradient.shape)
        with torch.no_grad():
            for i in range(d):
                sum1[i] += gradient[i] * scorenet1(x)[i]
                # sum3[i] += px * fx_grad[i]

    for i in range(int(N_query/batch_size)):
        if i % 100 == 0:
            print("number,", i)
        x = X_train[i * batch_size: (i + 1) * batch_size]

        px = torch.sum(torch.exp(true_gmm_model._estimate_log_prob(x) + torch.log(true_gmm_model.pi)), dim=1).squeeze() #[batch_size]
        # print("px, ", px.shape)

        fx_grad = []
        score_model = torch.sum(scorenet1(x), dim=0)#[d]
        # print("score_model,", score_model.shape)
        for i in range(d):
            fx_grad.append(autograd.grad(score_model[i], x, create_graph=True)[0][:, i]) #[d, batch_size]
        # print("fx-grad,", len(fx_grad))
        # print("fx_grad[0]", fx_grad[0].shape)

        with torch.no_grad():
            for i in range(d):
                sum2[i] += torch.sum(px * fx_grad[i])

    print("sum1", sum1)
    # print("sum2", -sum3)
    print("sum2", -sum2)
    print(torch.sum(sum1))
    # print(torch.sum(-sum3))
    print(torch.sum(-sum2))
    print("compare", torch.sum(sum1)/torch.sum(-sum2))

if __name__ == '__main__':
    maintest_()
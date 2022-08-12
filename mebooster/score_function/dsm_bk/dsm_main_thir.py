import argparse
import os

import yaml
from sympy import *
from torch.utils.data import Dataset, DataLoader

from dsm_bk.dsm_loss import anneal_dsm_sec_score_estimation
from gmm import GaussianMixture
from datetime import datetime
from math import pi

from ncsnv2 import get_sigmas, NCSNv2Simple, DSM_Generator_S2
from score_utils import *
import torch.optim as optim

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
    # diff_1 = diff(px, x_m, 1)
    diff_2 = diff(px, x_m, 2).reshape(d, d)
    # print("diff2.shape", diff_2.shape)
    # print(datetime.now())
    # diff_3 = diff(diff_2, x_m, 1).reshape(d, d, d)
    # print("diff3.shape", diff_3.shape)
    # print(datetime.now())
    # print("px,", px)
    s2_gradient = []
    # s3_gradient = []
    for j in range(d):
        s2_gradient.append(lambdify((x_s), diff_2[j, :], "numpy"))
    #     s3_gradient.append(lambdify((x_s), diff_3[j, :, :], "numpy"))
    #     # for k in range(d):
    #     #     s3_gradient.append(lambdify((x_s), diff_3[j, k, :], "numpy"))
    #     print("j", j)
    #     print(datetime.now())

    # s1_gradient = lambdify((x_s), diff_1, "numpy")
    S2 = torch.zeros([N_query, d, d]).to(device)

    px_func = lambdify((x_s), px, "numpy")
    print("start train query")
    # S1 = torch.zeros([N_query, d]).to(device)
    for i in range(N_query):
        if i % 100 == 0:
            print("number,", i)
            print(datetime.now())
        x = X_train[i].cpu().detach().numpy().tolist()
        # print("x,", x.shape)
        # s1_g = torch.zeros([d]).to(device)
        s2_g = torch.zeros([d, d]).to(device)
        # s3_g = torch.zeros([d, d, d]).to(device)

        # s1_g = torch.tensor(s1_gradient(*x)).squeeze().to(device)
        for j in range(d):
          s2_g[j, :] = torch.tensor(s2_gradient[j](*x)).to(device)
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
            S2[i] = s2_g / px_0
            # S1[i] = s1_g / px_0 #(-1) *

    return S2.view(N_query, d**2) #S2, S3

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

    # for i in range(len(s1_1)):
    #     # print(scores1[i] - rand1[i])
    #     error += torch.sum(torch.pow(s1_1[i] + s1_2[i], 2))
    # print("plus_error,", error/len(s1_1))
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

def train_denoise_sec_score(dataload, scorenet2, score_opt, epochs, sigmas, testx, scores_test):
    scorenet2.train()
    step=0

    for epoch in range(epochs):
        for data, target in dataload:
            step += 1
            score_opt.zero_grad()
            dsm_loss = anneal_dsm_sec_score_estimation(scorenet2, data, sigmas, None)
            dsm_loss.backward()
            score_opt.step()
        if epoch % 10==0:
            print("[{}]dsm_loss".format(epoch), dsm_loss)
        if epoch %10 == 0:
            # test_labels = torch.randint(0, len(sigmas), (testx.shape[0],), device=data.device)
            # print(scorenet1(data[0:5], test_labels))
            # print('scorenet[0:5]', scorenet2(testx))
            eval_l2_distance(scorenet2(testx), scores_test)
    return scorenet2

def get_optimizer(config, parameters):
    if config.optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                          betas=(config.optim.beta1, 0.999), amsgrad=config.optim.amsgrad,
                          eps=config.optim.eps)
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} not understood.'.format(config.optim.optimizer))


def get_model(config):
    if config.data.dataset == 'CIFAR10' or config.data.dataset == 'MNIST':
        return NCSNv2(config).to(config.device)
    else:
        return NCSNv2Simple(config).to(config.device)

def main():
    # parse config file
    with open(os.path.join('../configs', 'dsm.yml'), 'r') as f:
        pre_config = yaml.full_load(f)
    config = dict2namespace(pre_config)

    #args
    device = 'cuda:0'
    d = config.data.dim#10 #config/dsm.yml data:dim should also be the same
    N_query = 20000#50000
    n_com = 2#5
    batch_size=32

    #data
    print("data_generate")
    x, gt_mu, gt_var = generate_gmm_data(N_query, d, n_com, device) #
    y = torch.randn([N_query, 1]).to(device)
    dataset = TransferSetGaussian(x, y)
    dataload = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # true_gmm_model = GaussianMixture(n_components=n_com, n_features=d).to(device)
    # true_gmm_model.mu.data[0] = torch.tensor(gt_mu).to(device)
    # true_gmm_model.var.data[0][:] = torch.diag(torch.ones([d])).to(device)
    # true_gmm_model.pi.data[0] = (torch.ones([d, 1])/d).to(device)

    # print(true_gmm_model.mu.data[0]) #1, 10, 10
    # print(true_gmm_model.var[0]) #1, 10, 10, 10
    # print(true_gmm_model.pi.data[0]) #1, 10, 1

    # true_scores1 = calculate_score_functions_sympy(x, true_gmm_model, N_query,  d, device)
    # x.requires_grad = True
    # true_scores1 = calculate_score_functions(x, true_gmm_model, N_query, d, device)
    #gmm_model

    print("gmm_model")
    # used_sigmas = 3
    # noise = (torch.randn_like(x) * used_sigmas).to(device)
    gmm_model = fit_gaussian(x, n_com, d, device)
    print("calculate_score")
    scores2 = calculate_score_functions_sympy(x, gmm_model, N_query, d, device) #scores2, scores3

    # score = get_model(config)
    # s_model1 = torch.nn.DataParallel(score)
    # optimizer = get_optimizer(config, s_model1.parameters())
    scorenet2 = DSM_Generator_S2(config).to(device) #NCSNv2SimpleS2

    sigmas = get_sigmas(config)
    # print("sigmas", sigmas)
    # labels = torch.randint(0, len(sigmas), (x.shape[0],), device=x.device)
    s_model2 = scorenet2(x)

    # print("scores2", s_model2[0: 5])
    print("init_model_score vs sympy_score")
    eval_l2_distance(s_model2, scores2)
    score_opt = torch.optim.Adam(scorenet2.parameters(), lr=config.optim.lr)
    scorenet2 = train_denoise_sec_score(dataload, scorenet2, score_opt, config.training.n_epochs, sigmas, x[0: 10], scores2[0: 10])
    torch.save(scorenet2.state_dict(), '../model/scorenet2.tar.pth')

    labels = torch.randint(0, len(sigmas), (x.shape[0],), device=x.device)
    s_model2 = scorenet2(x)
    # print("init_model_score vs sympy_score")
    eval_l2_distance(s_model2, scores2)
    # # scorenet2 =
    # # scorenet3 =
    # s_model1 = scorenet1(x)
    # # s_model2 = scorenet2(x)
    # # s_model3 = scorenet3(x)
    rand2 = torch.randn([N_query, d**2]).to(device)
    # # rand2 = torch.randn([d,d]).to(device)
    # # rand3 = torch.randn([d,d,d]).to(device)
    #
    # print("eval_l2_distance")
    print("rand_score vs sympy_score")
    eval_l2_distance(scores2, rand2)
    print("trained_model_score vs sympy_score")
    eval_l2_distance(s_model2, scores2)
    # print("rand1", rand2[0:5])
    print("s_model2", s_model2[0:5])
    print("scores2", scores2[0:5])
    torch.save(x, '../model/x2.pt')
    torch.save(scores2, '../model/scores2.pt')
    torch.save(s_model2, '../model/dsm_model2.pt')

if __name__ == '__main__':
    main()
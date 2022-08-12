import torch
from torch import nn, optim
from torch.autograd import Variable

import lr_scheduler
import torch.nn.functional as F
import os.path as osp
import numpy as np
import torchvision

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

class ConvSparseNet(nn.Module):
    def __init__(self, d, ny):
        super().__init__()
        # self.z_dim = config.model.z_dim
        # self.register_buffer('sigmas', get_sigmas(config))
        hidden_units = 128
        self.d = d #16
        self.ny = ny #10
        self.conv1 = nn.Conv2d(1, 10, 4, 3)#16-4 /3--5, 10-4 /3--3
        self.conv2 = nn.Conv2d(10, 10, 3, 1)# 3*1
        self.main = nn.Sequential(
           nn.Linear(int(10 * 3), hidden_units),
           nn.Softplus(),
           nn.Linear(hidden_units, hidden_units),
           nn.Softplus(),
           nn.Linear(hidden_units, hidden_units),
           nn.Softplus(),
           nn.Linear(hidden_units, self.ny))

    def forward(self, x):
        x = x.view(1, 1, self.d, self.ny)
        # Xz = torch.cat([X, z], dim=-1)
        x = F.softplus(self.conv1(x))
        x = F.softplus(self.conv2(x))
        x = x.view(1, -1)
        output = self.main(x)
        # used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))
        # output = output / used_sigmas
        return output

class ConvSparseNet1(nn.Module):
    def __init__(self, d, ny):
        super().__init__()
        # self.z_dim = config.model.z_dim
        # self.register_buffer('sigmas', get_sigmas(config))
        hidden_units = 128
        self.d = d #16
        self.ny = ny #10
        self.conv1 = nn.Conv2d(1, 10, 4, 4)
        self.main = nn.Sequential(
           nn.Linear(int(10 * 5), hidden_units),
           nn.Softplus(),
           nn.Linear(hidden_units, hidden_units),
           nn.Softplus(),
           nn.Linear(hidden_units, hidden_units),
           nn.Softplus(),
           nn.Linear(hidden_units, self.ny))

    def forward(self, x):
        x = x.view(1, 1, self.d, self.y)
        # Xz = torch.cat([X, z], dim=-1)
        x = F.softplus(self.conv1(x))
        x = x.view(1, -1)
        output = self.main(x)
        # used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))
        # output = output / used_sigmas
        return output

class SparseNet(nn.Module):
    def __init__(self, d, ny):
        super().__init__()
        # self.z_dim = config.model.z_dim
        # self.register_buffer('sigmas', get_sigmas(config))
        hidden_units = 128
        self.d = d
        self.ny = ny
        self.main = nn.Sequential(
           nn.Linear(int(self.d * self.ny), hidden_units),
           nn.Softplus(),
           nn.Linear(hidden_units, hidden_units),
           nn.Softplus(),
           nn.Linear(hidden_units, hidden_units),
           nn.Softplus(),
           nn.Linear(hidden_units, self.ny))

    def forward(self, x):
        x = x.view(1, -1)
        # Xz = torch.cat([X, z], dim=-1)
        output = self.main(x)
        # used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))
        # output = output / used_sigmas
        return output

def conditional_optimization(M, ej, d, ny):
    # w_o = torch.zeros_like(w)
    # solve min_w ||w^TM||1 subject to (Mej)^T w=1,
    sparsenet = SparseNet(d, ny).to('cpu')#CovSparseNet
    sparsenet.train()
    optimizer = get_optimizer(sparsenet.parameters(), 'sgd', lr=0.01, momentum=0.5)#0.0005
    scheduler = lr_scheduler.lr_scheduler(mode='cos',
                                          init_lr=0.01,#0.000005
                                          num_epochs=8000,
                                          iters_per_epoch=1,
                                          lr_milestones=[3000],
                                          lr_step_multiplier=0.5,
                                          slow_start_epochs=2,
                                          slow_start_lr=0.01,
                                          end_lr=0.00001,
                                          multiplier=1,
                                          decay_factor=0.97,
                                          decay_epochs=0.8,
                                          staircase=True
                                         )
    for i in range(50):
        optimizer.zero_grad()
        w = sparsenet(M).view(ny, 1)
        scheduler(optimizer, i, i)
        loss = torch.sum(torch.abs(torch.matmul(w.T, M))) + 100 * torch.abs((torch.matmul(torch.matmul(M, ej).T, w) - 1.0))
        if i % 4000 == 0:
            print("epoch", i, ",loss:", loss.data.numpy())
        loss.backward(retain_graph=True)
        optimizer.step()
    return w


def sparse_dictionary_learning(M, d, m1, ny, lamb=0.001):
    """
    Args:
        M: the moment, \in ny \times d
        d: the size of the kernel/ the length of the neuron
        m1: the number of kernels/neurons
        ny: the length of the output vector
        lamb: the threshold for sparse filtering
    Returns: the estimated parameters, m1 \times d
    """
    A = torch.randn([m1, d])

    S = torch.zeros([d, d])
    S0 = torch.zeros([d])

    M = M.cpu().float()
    for dim in range(d):
        print("dim,", dim)
        #solve min_w ||w^TM||1 subject to (Mej)^T w=1,
        ej = torch.zeros([d, 1])
        ej[dim, 0] = 1.
        #w = torch.randn([ny, 1])
        w_o = conditional_optimization(M, ej, d, ny)
        s = torch.matmul(w_o.T, M) # 1 x d
        S[dim, :] = s
        S0[dim] = torch.sum(torch.abs(s) > lamb)

    if m1 >d:
        m1=d
    for k in range(m1):
        #S0
        l = torch.argmin(S0)
        v = S[l]
        A[k, :] = v
        S0[l] = d+1
    return A.T

class dictionary(nn.Module):
    """
    Class which defines an mxn linear dictionary.
    """

    def __init__(self, out_size, in_size,
                 datName='noname', use_cuda=True, useBias=False):
        super(dictionary, self).__init__()
        self.atoms = nn.Linear(in_size, out_size, bias=useBias)
        if use_cuda:
            self.atoms = self.atoms.cuda()
        self.m = out_size
        self.n = in_size
        self.datName = datName
        self.use_cuda = use_cuda

    # Set Dictionary Weights:
    def setWeights(self, weights):
        self.atoms.weight.data = weights

    # Scale Dictionary Weights:
    def scaleWeights(self, num):
        self.atoms.weight.data *= num

    #######################
    ## BASIC OPERATIONS
    #######################
    # Forward Pass (decoding)
    def forward(self, inputs):
        return self.atoms(inputs)

    # Transpose Pass ([roughly] encoding)
    def encode(self, inputs):
        return F.linear(self.atoms.weight.t(), inputs).t()

    # This worked:
    #        return torch.matmul(self.atoms.weight.t(), input.t()).t()

    # Normalize each column (a.k.a. atom) for the dictionary
    def normalizeAtoms(self):
        """
        Normalize each column to ||a||=1.
        """
        for a in range(0, self.n):
            atom = self.atoms.weight.data[:, a]
            aNorm = atom.norm()
            atom /= (aNorm + 1e-8)
            self.atoms.weight.data[:, a] = atom

    # Find Maximum Eigenvalue using Power Method
    def getMaxEigVal(self, iters=20):
        """
        Find Maximum Eigenvalue using Power Method
        """
        with torch.no_grad():
            bk = torch.ones(1, self.n)
            if self.use_cuda:
                bk = bk.cuda()

            for n in range(0, iters):
                f = bk.abs().max()
                bk = bk / f
                bk = self.encode(self.forward(bk))
            self.maxEig = bk.abs().max().item()

    # Return copies of the weights
    def getDecWeights(self):
        return self.atoms.weight.data.clone()

    def getEncWeights(self):
        return self.atoms.weight.data.t().clone()

    #######################
    ## VISUALIZATION
    #######################
    # Print the weight values
    def printWeightVals(self):
        print(self.getDecWeights())

    def printAtomImage(self, filename):
        imsize = int(np.sqrt(float(self.m)))
        # Normalize.
        Z = self.getDecWeights()
        #        Z = Z = Z - Z.min()
        #        Z = Z/(Z.abs().max())
        W = torch.Tensor(self.n, 1, imsize, imsize)
        for a in range(self.n):
            W[a][0] = Z[:, a].clone().resize_(imsize, imsize)
        # Number of atom images per row.
        nr = int(np.sqrt(float(self.n)))
        torchvision.utils.save_image(W, filename, nrow=nr,
                                     normalize=True, pad_value=255)

class soft_thresh(nn.Module):

    def __init__(self, lams, n_list=[], use_cuda=False):
        super(soft_thresh, self).__init__()

        self.use_cuda = use_cuda
        self.shrink = []
        self.n_list = n_list

        if type(lams) == float:
            self.lams = [lams]
            self.shrink.append(nn.Softshrink(lams))

        elif type(lams) == list:
            self.lams = lams
            for lam in lams:
                self.shrink.append(nn.Softshrink(lam))

        if use_cuda:
            for s, shr in enumerate(self.shrink):
                self.shrink[s] = shr.cuda()

    #        soft_thresh.register_forward_pre_hook(nothing)

    def forward(self, input):

        if len(self.lams) == 1:
            return self.shrink[0](input)
        else:
            outputs = []
            beg = 0
            n_d = 0
            for t, shrink in enumerate(self.shrink):
                beg += n_d
                n_d = self.n_list[t]
                outputs.append(shrink(input.narrow(1, beg, n_d)))
            return torch.cat(outputs, 1)

def FISTA(y0, A, m1, d, alpha, maxIter,
          returnCodes=True, returnCost=False, returnResidual=False):
    if not hasattr(A, 'maxEig'):
        A.getMaxEigVal()
    # shrink = soft_thresh(alpha / A.maxEig, [], A.cuda)
    returnTab = {}

    # INITIALIZE:
    batchSize = y0.size(0)
    # print("y0", y0.size()) #[9, 10], d, n
    # yk = Variable(torch.zeros(y0.size()))
    # xprev = Variable(torch.zeros(y0.size()))
    yk = Variable(torch.zeros([d, m1])) #9, 18
    xprev = Variable(torch.zeros([d, m1])) #9, 18
    if A.use_cuda:
        yk = yk.cuda()
        xprev = xprev.cuda()
    t = 1
    residual = A.forward(yk) - y0 #d, n
    # print("residual", residual)

    # TMP:
    cost = torch.zeros(maxIter)

    for it in range(0, maxIter):
        # ista step:
        # print("it", it)
        tmp = yk - A.encode(residual) / A.maxEig
        # xk = shrink.forward(tmp)
        xk = tmp

        # fista stepsize update:
        tnext = (1 + (1 + 2 * (t ** 2)) ** .5) / 2#(1 + (1 + 4 * (t ** 2)) ** .5) / 2
        fact = (t - 1) / tnext
        yk = xk + (xk - xprev) * fact

        # copies for next iter
        xprev = xk
        t = tnext
        residual = A.forward(yk) - y0
        # print("yk,", yk)
        # print("xk", xk)

        # compute any updates desired: (cost hist, code est err, etc.)
        # comphistories(it,yk, params, options, returntab)

        # if returnCost:
        #     fidErr = residual.data[0].norm() ** 2
        #     L1err = yk.data[0].abs().sum()
        #     cost[it] = float(fidErr + alpha * L1err) / batchSize

    # if maxIter == 0:
        # yk = shrink(A.enc(y0))

    if returnCodes:
        returnTab["codes"] = yk
    if returnResidual:
        returnTab["residual"] = residual
    if returnCost:
        returnTab["costHist"] = cost.numpy()

    return returnTab

import gc
def sparse_dictionary_learning_lista(M, d, m1_0, ny, lamb=0.01):
    m1 = m1_0
    #M [ny, d]
    #x [d]
    #X [m1, d]
    #A [ny, m1]
    sigLen = m1
    codeLen = ny
    Dict = dictionary(codeLen, sigLen)#ny, m1
    # Optimizer
    OPT = torch.optim.SGD(Dict.parameters(), lr=0.00001) #=0.000001
    # For learning rate decay
    scheduler = torch.optim.lr_scheduler.StepLR(OPT, step_size=30,
                                                gamma=1)  # 30, 1
    maxEpoch = 50 #20
    l1w = 0.5 #0.05 #it's not depend
    fistaIters = 20 #20
    # FISTA:
    fistaOptions = {"returnCodes": True,
                    "returnCost": False,
                    "returnFidErr": False}
    mseLoss = nn.MSELoss()
    mseLoss.size_average = True
    useL1Loss = True
    if useL1Loss:
        l1Loss = Variable(torch.FloatTensor(1), requires_grad=True)
    else:
        l1Loss = 0

    for it in range(maxEpoch):
        epochLoss = 0
        epoch_sparsity = 0
        epoch_rec_error = 0
        # ================================================
        # TRAINING

        Y = Variable((M.T).float()) #[d, ny]
        gc.collect()

        # print("Y.shape", Y.shape)
        ## CODE INFERENCE
        fistaOut = FISTA(Y, Dict, m1, d, l1w, fistaIters, fistaOptions)
        X = fistaOut["codes"]
        # print("X.shape", X.shape)
        gc.collect()

        ## FORWARD PASS
        Y_est = Dict.forward(X)  # try decoding the optimal codes
        # print("Y_est", torch.sum(Y_est.T - torch.matmul(Dict.atoms.weight, X.T)))
        # loss
        reconErr = mseLoss(Y_est, Y)
        if useL1Loss:
            l1Loss = Y_est.norm(1) / X.size(0)

        ## BACKWARD PASS
        batchLoss = reconErr + l1w * l1Loss
        batchLoss.backward()
        OPT.step()
        scheduler.step()
        Dict.zero_grad()
        Dict.normalizeAtoms()
        del Dict.maxEig

        ## Housekeeping
        sampleLoss = batchLoss.item()  # .data[0]
        epochLoss += sampleLoss

        sample_rec_error = reconErr.item()  # .data[0]
        epoch_rec_error += sample_rec_error

        sample_sparsity = ((X.data == 0).sum()) / X.numel()
        epoch_sparsity += sample_sparsity

        ## SANITY CHECK:
        # If the codes are practically all-zero, stop fast.
        # TODO: something smarter to do here? Lower L1 weight?
        if torch.abs(sample_sparsity - 1.0) < 0.001:
            print("CODES NEARLY ALL ZERO. SKIP TO NEXT EPOCH.")
            break

        ## Print stuff.
        # You may wish to print some figures here too. See bottom of page.
        if it % 50 == 0:
            print('Train Epoch: {}]'.format(it))
            print('Loss: {:.6f} \tRecon Err: {:.6f} \tSparsity: {:.6f} '.format(
                sampleLoss, sample_rec_error, sample_sparsity))

        ## end "TRAINING" batch-loop
        # ================================================

        ## need one for training, one for testing
        epoch_average_loss = epochLoss / 1
        epoch_avg_recErr = epoch_rec_error / 1
        epoch_avg_sparsity = epoch_sparsity / 1

        #        lossHist[it]  = epoch_average_loss
        #        errHist[it]   = epoch_avg_recErr
        #        spstyHist[it] = epoch_avg_sparsity
        if it % 50 == 0:
            print('- - - - - - - - - - - - - - - - - - - - -')
            print('EPOCH ', it + 1, '/', maxEpoch, " STATS")
            print('LOSS = ', epoch_average_loss)
            print('RECON ERR = ', epoch_avg_recErr)
            print('SPARSITY = ', epoch_avg_sparsity, '\n')
    ## end "EPOCH" loop
    dir = X.T
    # print("X.T", X.T)
    dir = dir/torch.norm(dir, dim=1).view(-1, 1)
    print("dir.shape", dir.shape)
    return dir.T

import gc
def sparse_dictionary_learning_lista(M, d, m1_0, ny, lamb=0.01):
    m1 = m1_0
    #M [ny, d]
    #x [d]
    #X [m1, d]
    #A [ny, m1]
    sigLen = m1
    codeLen = ny
    Dict = dictionary(codeLen, sigLen)#ny, m1
    # Optimizer
    OPT = torch.optim.SGD(Dict.parameters(), lr=0.001) #=0.000001
    # For learning rate decay
    scheduler = torch.optim.lr_scheduler.StepLR(OPT, step_size=30,
                                                gamma=1)  # 30, 1
    maxEpoch = 20 #20
    l1w = 0.05 #0.05 #it's not depend
    fistaIters = 20 #20
    # FISTA:
    fistaOptions = {"returnCodes": True,
                    "returnCost": False,
                    "returnFidErr": False}
    mseLoss = nn.MSELoss()
    mseLoss.size_average = True
    useL1Loss = False#True
    if useL1Loss:
        l1Loss = Variable(torch.FloatTensor(1), requires_grad=True)
    else:
        l1Loss = 0

    for it in range(maxEpoch):
        epochLoss = 0
        epoch_sparsity = 0
        epoch_rec_error = 0
        # ================================================
        # TRAINING

        Y = Variable((M.T).float()) #[d, ny]
        gc.collect()

        # print("Y.shape", Y.shape)
        ## CODE INFERENCE
        fistaOut = FISTA(Y, Dict, m1, d, l1w, fistaIters, fistaOptions)
        X = fistaOut["codes"]
        # print("X.shape", X.shape)
        gc.collect()

        ## FORWARD PASS
        Y_est = Dict.forward(X)  # try decoding the optimal codes
        # print("Y_est", torch.sum(Y_est.T - torch.matmul(Dict.atoms.weight, X.T)))
        # loss
        reconErr = mseLoss(Y_est, Y)
        if useL1Loss:
            l1Loss = Y_est.norm(1) / X.size(0)

        ## BACKWARD PASS
        batchLoss = reconErr + l1w * l1Loss
        batchLoss.backward()
        OPT.step()
        scheduler.step()
        Dict.zero_grad()
        Dict.normalizeAtoms()
        del Dict.maxEig

        ## Housekeeping
        sampleLoss = batchLoss.item()  # .data[0]
        epochLoss += sampleLoss

        sample_rec_error = reconErr.item()  # .data[0]
        epoch_rec_error += sample_rec_error

        sample_sparsity = ((X.data == 0).sum()) / X.numel()
        epoch_sparsity += sample_sparsity

        ## SANITY CHECK:
        # If the codes are practically all-zero, stop fast.
        # TODO: something smarter to do here? Lower L1 weight?
        if torch.abs(sample_sparsity - 1.0) < 0.001:
            print("CODES NEARLY ALL ZERO. SKIP TO NEXT EPOCH.")
            break

        ## Print stuff.
        # You may wish to print some figures here too. See bottom of page.
        if it % 50 == 0:
            print('Train Epoch: {}]'.format(it))
            print('Loss: {:.6f} \tRecon Err: {:.6f} \tSparsity: {:.6f} '.format(
                sampleLoss, sample_rec_error, sample_sparsity))

        ## end "TRAINING" batch-loop
        # ================================================

        ## need one for training, one for testing
        epoch_average_loss = epochLoss / 1
        epoch_avg_recErr = epoch_rec_error / 1
        epoch_avg_sparsity = epoch_sparsity / 1

        #        lossHist[it]  = epoch_average_loss
        #        errHist[it]   = epoch_avg_recErr
        #        spstyHist[it] = epoch_avg_sparsity
        if it % 50 == 0:
            print('- - - - - - - - - - - - - - - - - - - - -')
            print('EPOCH ', it + 1, '/', maxEpoch, " STATS")
            print('LOSS = ', epoch_average_loss)
            print('RECON ERR = ', epoch_avg_recErr)
            print('SPARSITY = ', epoch_avg_sparsity, '\n')
    ## end "EPOCH" loop
    dir = X.T
    # print("X.T", X.T)
    dir = dir/torch.norm(dir, dim=1).view(-1, 1)
    print("dir.shape", dir.shape)
    return dir.T

#QR based Jacbi-like JD; This function minimizes the cost
import torch
import numpy as np

def qrj1d(X, device): #varargin
    n, m =X.shape #p, p*L(L=2p)
    N = int(m/n);
    BB = []
    #default values
    EERRJ2 = 0
    ERR = 1e-4
    RBALANCE=3
    ITER=1000 #iteration
    MODE = 'B'
    #no nargin

    Err = ERR
    Rbalance = RBALANCE
    X1 = X
    B = torch.eye(n).to(device) #nxn
    # Binv = torch.eye(n)
    J = 0

    for t in range(N):
        J = J + torch.square(torch.norm(X1[:, t*n: (t+1)*n] - torch.diag(torch.diag(X[:, t*n : (t+1)*n])))) #Frobenius 范数。
    JJ0 = J #.cpu().detach().numpy() # JJ=[JJ, J]

    #the following part implements a sweep
    err = ERR*n+1
    if MODE == 'B':
       ERR = ERR*n
    # print("X first time", X)
    k=0
    while err > ERR and k < ITER:
        k=k+1
        L = torch.eye(n).to(device)
        U = torch.eye(n).to(device)
        # Dinv = torch.eye(n)

        for i in range(1, n):
            for j in range(i): #(i-1)?
                # print("i", i)
                # print("j", j)
                G = torch.tensor(np.array([(-X[i, range(i, m, n)] + X[j, range(j, m, n)]).cpu().detach().numpy(),
                                  (-2 * X[i, range(j, m, n)]).cpu().detach().numpy()])).squeeze().to(device)
                U1, D1, V1 = torch.svd(torch.matmul(G, G.T))#no int
                v = U1[:, 0]
                tetha = 1 / 2 * torch.atan(v[1] / v[0])
                c = torch.cos(tetha)
                s = torch.sin(tetha)
                h1 = c * X[:, range(j, m, n)] - s * X[:, range(i, m, n)]
                h2 = c * X[:, range(i, m, n)] + s * X[:, range(j, m, n)]
                X[:, range(j, m, n)] = h1
                X[:, range(i, m, n)] = h2
                h1 = c * X[j, :] - s * X[i, :]
                h2 = s * X[j, :] + c * X[i, :]
                X[j, :] = h1
                X[i, :] = h2
                h1 = c * U[j, :] - s * U[i, :]
                h2 = s * U[j, :] + c * U[i, :]
                U[j, :] = h1
                U[i, :] = h2
            # print("X sec", X)

        for i in range(n):
            # rindex = []
            # Xj = []
            for j in range(i+1, n):
                cindex = list(range(m))
                # remove range(j,m,n)
                for ele in range(j, m, n):
                   cindex.remove(ele)
                a = -torch.matmul(X[i, cindex], X[j, cindex].T) / torch.matmul(X[i, cindex], X[i, cindex].T).squeeze()
                if torch.abs(a) > 1:
                    a = torch.sign(a)
                X[j, :] = a * X[i, :] + X[j, :]
                I = list(range(i, m, n))
                J = list(range(j, m, n))
                X[:, J] = a * X[:, I] + X[:, J]
                L[j, :] = L[j, :] + a * L[i, :]
        # print("B", B)
        # print("X", X)
        B = torch.matmul(torch.matmul(L, U), B)
        # print("B", B)
        err = torch.max(torch.max(torch.abs(torch.matmul(L, U) - torch.eye(n).to(device)), dim=0).values)
        EERR = err.clone().detach()  #[EERR, err]
        if k % RBALANCE == 0:
            d = torch.sum(torch.abs(X.T), dim=0)#.float()
            D = torch.diag(1 / (N * d))  # here i assume N is in the lower
            # print("D.diag", D)
            for t in range(N):
                X[:, (t * n):((t + 1) * n)] = torch.matmul(torch.matmul(D, X[:, (t * n):((t + 1) * n)]), D)
            B = torch.matmul(D, B)
            # print('D', D)
        J = 0
        # print("B", B.shape)
        BB.append(B.cpu().detach().numpy())#BB[:, :, k] = B
        # print("B", B)
        Binv = torch.inverse(B)
        for t in range(N):
            # print("Binv", Binv.shape)
            # print("diag", torch.diag(torch.diag(X[:, t*n:(t+1)*n])).shape)
            J0_1 = torch.matmul(Binv, torch.diag(torch.diag(X[:, t*n:(t+1)*n]))) #n, n
            J0_2  = torch.matmul(J0_1, Binv.T)
            J0 = torch.norm(X1[:, t*n:(t+1)*n] - J0_2)
            J = J + J0 * J0
        JJ = torch.tensor([0, 0]).to(device)
        JJ[0] = JJ0
        JJ[1] = J
        if MODE =='E':
            err = torch.abs(JJ[-2]-JJ[-1])/JJ[-1]
            EERRJ2 = err
        # print("X", X[0,:])
        # print("B", B)
        # print("k", k)
        if k % 100 == 0:
            print("qrj1d-err", err)
    Y = X
    S = {'iterations': k,
         'LUerror': EERR,
         'J2error': JJ,
         'J2RelativeError': EERRJ2
        }

    varargout = [S, torch.tensor(np.array(BB)).to(device)]
    return Y, B, varargout
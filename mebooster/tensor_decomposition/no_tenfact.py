import torch

from qrj1d import qrj1d


def ttm(T, d, V, deivce):
    T1 = torch.zeros([T.shape[0], T.shape[1]]).to(deivce)
    for i in range(d):
        T1 = T1 + V[i]*T[:, :, i]
    return T1


def no_tenfact(T, L, k, device):
                 #L=2*p
    p = T.shape[0]
    sweeps = [0., 0.]

    #stage 1, random projections
    M = torch.zeros([p, p*L]).to(device)
    W = torch.zeros([p, L]).to(device)

    for l in range(L):
        W[:, l] = torch.randn([p]).to(device)
        W[:, l] = W[:, l]/torch.norm(W[:, l])
        # print("W[:,l]", W[:,l])
        # print("norm", torch.norm(W[:,l]))
        M[:, l*p:(l+1)*p] = ttm(T, p, W[:, l], device)
    # print("M", M)

    D, U, S = qrj1d(M, device)
    #calculate the true eigenvalues across all matrices
    Ui = torch.inverse(U)
    Ui_norms = torch.sqrt(torch.sum(Ui*Ui, dim=0)).view(1, -1) #a row
    Ui_normlalized = Ui/Ui_norms
    dot_products = torch.matmul(Ui_normlalized.T, W)
    Lambdas = torch.zeros([p, L])
    for l in range(L):
        lam_temp = (torch.diag(D[:, l*p:((l+1)*p)])/(dot_products[:, l])).view(1, -1)
        # print("lam_temp", lam_temp.shape)
        # print("Ui_norms", Ui_norms.shape)
        Lambdas[:,l] = (lam_temp * (Ui_norms*Ui_norms)).view(1, -1)

    #calculate the best eigenvalues and eigenvectors
    _, idx0 = torch.sort(torch.mean(torch.abs(Lambdas), dim=1), dim=0, descending=True)
    # Lambda0 = torch.mean(Lambdas[idx0[:k],:], dim=1)
    V = Ui_normlalized[:, idx0[:k]]

    #store number of sweeps
    sweeps[0] = S[0]['iterations']
    sweeps[1] = S[0]['iterations']
    #stage 2: plugin projections
    W =Ui_normlalized
    M = torch.zeros([p, p*W.shape[1]]).to(device)

    for l in range(W.shape[1]):
        w = W[:, l]
        w = w/torch.norm(w)
        M[:, l*p:(l+1)*p] = ttm(T, p, w, device)

    D, U, S = qrj1d(M, device)
    Ui = torch.inverse(U)
    Ui_norm = Ui/torch.sqrt(torch.sum(Ui*Ui, dim=0)).view(1, -1)#torch.norm(Ui, dim=1)
    V1 = Ui_norm
    sweeps[1] = sweeps[1] + S[0]['iterations']

    Lambda = torch.zeros([p, 1]).to(device)
    for i in range(p):
        Z = torch.inverse(V1)
        X = torch.matmul(torch.matmul(Z, M[:, l*p:(l+1)*p]), Z.T)
        Lambda = Lambda + torch.abs(torch.diag(X))

    _, idx = torch.sort(torch.abs(Lambda), dim=0, descending=True)
    V1 = Ui_norm[:, idx[:k]]

    misc = {'V0': V,
        'sweeps': sweeps
        }
    return V1, Lambda, misc
"""
MLE for Gaussian Mixture Model using EM
"""
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import torch


class gmm():
    def __init__(self, X, N, K):
        # GMM params
        self.mus = np.array([[1, 1], [15, 15]], dtype='float')
        self.sigmas = np.array([np.diag([1, 1]), np.diag([1, 1])], dtype='float')
        self.pis = np.array([0.5, 0.5])
        self.X = X
        self.N = N
        self.K = K

    def fitting(self, iter_num=10):
        for it in range(iter_num):
            # E-step
            gammas = np.zeros([self.N, self.K])

            for k in range(self.K):
                lik = st.multivariate_normal.pdf(self.X, mean=self.mus[k], cov=self.sigmas[k])
                gammas[:, k] = self.pis[k] * lik

            # Evaluate
            loglik = np.sum(np.log(np.sum(gammas, axis=1)))
            print('Log-likelihood: {:.4f}'.format(loglik))
            print('Mus: {}'.format(self.mus))
            print()

            # Normalize gamma
            gammas = gammas / np.sum(gammas, axis=1)[:, np.newaxis]

            # M-step
            for k in range(self.K):
                Nk = np.sum(gammas[:, k])

                mu = 1 / Nk * np.sum(gammas[:, k][:, np.newaxis] * self.X, axis=0)

                Xmu = (self.X - mu)[:, :, np.newaxis]
                sigma = 1 / Nk * np.sum(
                    [gammas[i, k] * Xmu[i] @ Xmu[i].T for i in range(self.N)],
                    axis=0
                )

                pi = Nk / self.N

                self.mus[k] = mu
                self.sigmas[k] = sigma
                self.pis[k] = pi

    def px(self, x):
        p_x = 0.0
        for j in range(self.K):
            content = 2 * torch
            p_x = p_x + self.pis[j]
        return p_x

# Generate data
X1 = np.random.multivariate_normal([5, 5], np.diag([0.5, 0.5]), size=20)
X2 = np.random.multivariate_normal([8, 8], np.diag([0.5, 0.5]), size=20)
X = np.vstack([X1, X2])
N = X.shape[0]
K = 2
gmm =gmm(X=X, N=N, K=K)


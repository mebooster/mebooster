import torch
import math as m
import numpy as np
import random
import time

import torch
import torch.utils
import torch.utils.data
from torch import nn

import data
from vae import flow
import argparse
import pathlib

def add_args(parser):
    parser.add_argument("--latent_size", type=int, default=2)
    parser.add_argument("--variational", choices=["flow", "mean-field"], default="flow")
    parser.add_argument("--flow_depth", type=int, default=2)
    parser.add_argument("--data_size", type=int, default=10) #784 for mnist
    parser.add_argument("--learning_rate", type=float, default=0.00001)#0.001
    # parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--max_iterations", type=int, default=10000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=582838)
    parser.add_argument("--train_dir", type=pathlib.Path, default="./vae_tmp")
    parser.add_argument("--data_dir", type=pathlib.Path, default="./vae_tmp")

class NormalLogProb(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, loc, scale, z):
        var = torch.pow(scale, 2)
        return -0.5 * torch.log(2 * np.pi * var) - torch.pow(z - loc, 2) / (2 * var)


class BernoulliLogProb(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, target):
        # bernoulli log prob is equivalent to negative binary cross entropy
        return -self.bce_with_logits(logits, target)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        modules = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        ]
        self.net = nn.Sequential(*modules)

    def forward(self, input):
        return self.net(input)


class Model(nn.Module):
    """Variational autoencoder, parameterized by a generative network."""

    def __init__(self, latent_size, data_size):
        super().__init__()
        self.register_buffer("p_z_loc", torch.zeros(latent_size))
        self.register_buffer("p_z_scale", torch.ones(latent_size))
        self.log_p_z = NormalLogProb()
        self.log_p_x = BernoulliLogProb()
        self.generative_network = NeuralNetwork(
            input_size=latent_size, output_size=data_size, hidden_size=latent_size * 2
        )

    def forward(self, z, x):
        """Return log probability of model."""
        log_p_z = self.log_p_z(self.p_z_loc, self.p_z_scale, z).sum(-1, keepdim=True)
        logits = self.generative_network(z)
        # unsqueeze sample dimension
        logits, x = torch.broadcast_tensors(logits, x.unsqueeze(1))
        log_p_x = self.log_p_x(logits, x).sum(-1, keepdim=True)
        return log_p_z + log_p_x

@torch.no_grad()
def evaluate(n_samples, model, variational, eval_data):
    model.eval()
    total_log_p_x = 0.0
    total_elbo = 0.0
    for batch in eval_data:
        x = batch[0].to(next(model.parameters()).device)
        z, log_q_z = variational(x, n_samples)
        log_p_x_and_z = model(z, x)
        # importance sampling of approximate marginal likelihood with q(z)
        # as the proposal, and logsumexp in the sample dimension
        elbo = log_p_x_and_z - log_q_z
        log_p_x = torch.logsumexp(elbo, dim=1) - np.log(n_samples)
        # average over sample dimension, sum over minibatch
        total_elbo += elbo.cpu().numpy().mean(1).sum()
        # sum over minibatch
        total_log_p_x += log_p_x.cpu().numpy().sum()
    n_data = len(eval_data.dataset)
    return total_elbo / n_data, total_log_p_x / n_data

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class VariationalMeanField(nn.Module):
    """Approximate posterior parameterized by an inference network."""

    def __init__(self, latent_size, data_size):
        super().__init__()
        self.inference_network = NeuralNetwork(
            input_size=data_size,
            output_size=latent_size * 2,
            hidden_size=latent_size * 2,
        )
        self.log_q_z = NormalLogProb()
        self.softplus = nn.Softplus()

    def forward(self, x, n_samples=1):
        """Return sample of latent variable and log prob."""
        loc, scale_arg = torch.chunk(
            self.inference_network(x).unsqueeze(1), chunks=2, dim=-1
        )
        scale = self.softplus(scale_arg)
        eps = torch.randn((loc.shape[0], n_samples, loc.shape[-1]), device=loc.device)
        z = loc + scale * eps  # reparameterization
        log_q_z = self.log_q_z(loc, scale, z).sum(-1, keepdim=True)
        return z, log_q_z


class VariationalFlow(nn.Module):
    """Approximate posterior parameterized by a flow (https://arxiv.org/abs/1606.04934)."""

    def __init__(self, latent_size, data_size, flow_depth):
        super().__init__()
        hidden_size = latent_size * 2
        self.inference_network = NeuralNetwork(
            input_size=data_size,
            # loc, scale, and context
            output_size=latent_size * 3,
            hidden_size=hidden_size,
        )
        modules = []
        for _ in range(flow_depth):
            modules.append(
                flow.InverseAutoregressiveFlow(
                    num_input=latent_size,
                    num_hidden=hidden_size,
                    num_context=latent_size,
                )
            )
            modules.append(flow.Reverse(latent_size))
        self.q_z_flow = flow.FlowSequential(*modules)
        self.log_q_z_0 = NormalLogProb()
        self.softplus = nn.Softplus()

    def forward(self, x, n_samples=1):
        """Return sample of latent variable and log prob."""
        loc, scale_arg, h = torch.chunk(
            self.inference_network(x).unsqueeze(1), chunks=3, dim=-1
        )
        scale = self.softplus(scale_arg)
        eps = torch.randn((loc.shape[0], n_samples, loc.shape[-1]), device=loc.device)
        z_0 = loc + scale * eps  # reparameterization
        log_q_z_0 = self.log_q_z_0(loc, scale, z_0)
        z_T, log_q_z_flow = self.q_z_flow(z_0, context=h)
        log_q_z = (log_q_z_0 + log_q_z_flow).sum(-1, keepdim=True)
        return z_T, log_q_z
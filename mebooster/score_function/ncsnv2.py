import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
from functools import partial
# from . import get_sigmas
from layers import *
from normalization import get_normalization

def get_sigmas(config):
    if config.model.sigma_dist == 'geometric':
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(config.model.sigma_begin), np.log(config.model.sigma_end),
                               config.model.num_classes))).float().to(config.device)
    elif config.model.sigma_dist == 'uniform':
        sigmas = torch.tensor(
            np.linspace(config.model.sigma_begin, config.model.sigma_end, config.model.num_classes)
        ).float().to(config.device)

    else:
        raise NotImplementedError('sigma distribution not supported')

    return sigmas

class NCSNv2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.logit_transform = config.data.logit_transform
        self.rescaled = config.data.rescaled
        self.norm = get_normalization(config, conditional=False)
        self.ngf = ngf = config.model.ngf
        self.num_classes = num_classes = config.model.num_classes

        self.act = act = get_act(config)
        self.register_buffer('sigmas', get_sigmas(config))
        self.config = config

        self.begin_conv = nn.Conv2d(config.data.channels, ngf, 3, stride=1, padding=1)

        self.normalizer = self.norm(ngf, self.num_classes)
        self.end_conv = nn.Conv2d(ngf, config.data.channels, 3, stride=1, padding=1)

        self.res1 = nn.ModuleList([
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm),
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res2 = nn.ModuleList([
            ResidualBlock(self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res3 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm, dilation=2),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=2)]
        )

        if config.data.image_size == 28:
            self.res4 = nn.ModuleList([
                ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                              normalization=self.norm, adjust_padding=True, dilation=4),
                ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                              normalization=self.norm, dilation=4)]
            )
        else:
            self.res4 = nn.ModuleList([
                ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                              normalization=self.norm, adjust_padding=False, dilation=4),
                ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                              normalization=self.norm, dilation=4)]
            )

        self.refine1 = RefineBlock([2 * self.ngf], 2 * self.ngf, act=act, start=True)
        self.refine2 = RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, act=act)
        self.refine3 = RefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, act=act)
        self.refine4 = RefineBlock([self.ngf, self.ngf], self.ngf, act=act, end=True)

    def _compute_cond_module(self, module, x):
        for m in module:
            x = m(x)
        return x

    def forward(self, x, y):
        if not self.logit_transform and not self.rescaled:
            h = 2 * x - 1.
        else:
            h = x

        output = self.begin_conv(h)

        layer1 = self._compute_cond_module(self.res1, output)
        layer2 = self._compute_cond_module(self.res2, layer1)
        layer3 = self._compute_cond_module(self.res3, layer2)
        layer4 = self._compute_cond_module(self.res4, layer3)

        ref1 = self.refine1([layer4], layer4.shape[2:])
        ref2 = self.refine2([layer3, ref1], layer3.shape[2:])
        ref3 = self.refine3([layer2, ref2], layer2.shape[2:])
        output = self.refine4([layer1, ref3], layer1.shape[2:])

        output = self.normalizer(output)
        output = self.act(output)
        output = self.end_conv(output)

        used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))
        output = output / used_sigmas
        return output

class NCSN(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.logit_transform = config.data.logit_transform
        self.rescaled = config.data.rescaled
        self.norm = get_normalization(config, conditional=False)
        self.ngf = ngf = config.model.ngf
        self.input_dim = config.data.dim
        self.num_classes = num_classes = config.model.num_classes

        self.act = act = get_act(config)
        self.register_buffer('sigmas', get_sigmas(config))
        self.config = config

        self.normalizer = self.norm(ngf, self.num_classes)
        # self.end_conv = nn.Conv2d(ngf, config.data.channels, 3, stride=1, padding=1) #it's an (channel, width, height)

        self.res1 = nn.ModuleList([
            ResidualBlock(self.input_dim, self.ngf, resample=None, act=act,
                          normalization=self.norm),
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res2 = nn.ModuleList([
            ResidualBlock(self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm),
            ResidualBlock(2 * self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.refine1 = RefineBlock([2 * self.ngf], self.ngf, act=act, start=True)
        self.refine2 = RefineBlock([2 * self.ngf, self.ngf], self.input_dim, act=act)

    def _compute_cond_module(self, module, x):
        for m in module:
            x = m(x)
        return x

    def forward(self, x, y):
        # if not self.logit_transform and not self.rescaled:
        #     h = 2 * x - 1.
        # else:
        #     h = x

        # output = self.begin_conv(h)

        layer1 = self._compute_cond_module(self.res1, x)
        print('layer1', layer1.shape)
        layer2 = self._compute_cond_module(self.res2, layer1)
        print('layer2', layer2.shape)
        # layer3 = self._compute_cond_module(self.res3, layer2)
        # layer4 = self._compute_cond_module(self.res4, layer3)

        ref1 = self.refine1([layer2], layer2.shape[2:])
        print('ref', ref1.shape)
        # ref2 = self.refine2([layer3, ref1], layer3.shape[2:])
        # ref3 = self.refine3([layer2, ref2], layer2.shape[2:])
        output = self.refine2([layer1, ref1], layer1.shape[2:])
        print('output', output.shape)
        # output = self.normalizer(output)
        # output = self.act(output)
        # output = self.end_conv(output)

        used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))
        output = output / used_sigmas
        return output

    def test(self, x):

        layer1 = self._compute_cond_module(self.res1, x)
        layer2 = self._compute_cond_module(self.res2, layer1)
        # layer3 = self._compute_cond_module(self.res3, layer2)
        # layer4 = self._compute_cond_module(self.res4, layer3)

        ref1 = self.refine1([layer2], layer2.shape[2:])
        # ref2 = self.refine2([layer3, ref1], layer3.shape[2:])
        # ref3 = self.refine3([layer2, ref2], layer2.shape[2:])
        output = self.refine2([layer1, ref1], layer1.shape[2:])

        return output

class Score(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        nef = 16#config.model.nef
        self.u_net = nn.Sequential(
            # input is (nc) x 4 x 4
            nn.Conv2d(config.data.channels, nef, 2, stride=1, padding=1),
            # nn.Softplus(),
            nn.GroupNorm(4, nef),
            nn.ELU(),
            # state size. (nef) x 4 x 4
            nn.Conv2d(nef, nef * 2, 2, stride=1, padding=0),
            nn.GroupNorm(4, nef * 2),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef*2) x 2 x 2
            nn.Conv2d(nef * 2, nef * 4, 2, stride=1, padding=0),
            nn.GroupNorm(4, nef * 4),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef*4) x 1 x 1
            nn.ConvTranspose2d(nef * 4, nef * 2, 2, stride=2, padding=0),
            nn.GroupNorm(4, nef * 2),
            # nn.Softplus(),
            nn.ELU(),
            ## state size. (nef*2) x 2 x 2
            nn.ConvTranspose2d(nef * 2, nef, 2, stride=1, padding=1),
            nn.GroupNorm(4, nef),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef) x 3 x 3
            nn.ConvTranspose2d(nef, config.data.channels, 2, stride=1, padding=1),
            # nn.Softplus()
            nn.ELU()
            # state size. (nc) x 4 x 4
        )
        self.begin_fc = nn.Sequential(
            nn.Linear(config.data.dim, 16),
            nn.ELU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(config.data.channels * 4 * 4, 1024),
            nn.LayerNorm(1024),
            nn.ELU(),
            nn.Linear(1024, config.data.channels * 16)#*4*$
        )

    def forward(self, x):
        #x = self.begin_fc(x)
        x = x.reshape(-1, 1, 4, 4)
        #if x.is_cuda: and self.config.training.ngpu > 1:
        #    score = nn.parallel.data_parallel(
        #        self.u_net, x, list(range(self.config.training.ngpu)))
        #else:
        score = self.u_net(x)
        score = self.fc(score.view(x.shape[0], -1))
        #score = self.fc(score.view(x.shape[0], -1)).view(
        #    x.shape[0], self.config.data.channels, 28, 28)
        return score

class NCSNv2Simple(nn.Module):
    def __init__(self, d):
        super().__init__()
        # self.z_dim = config.model.z_dim
        # self.register_buffer('sigmas', get_sigmas(config))
        self.input_dim = d
        hidden_units = 4096#2048#1024
        self.main = nn.Sequential(
        nn.Linear(self.input_dim, hidden_units),
        nn.Softplus(),
        nn.Linear(hidden_units, hidden_units),
        nn.Softplus(),
        nn.Linear(hidden_units, hidden_units),
        nn.Softplus(),
        nn.Linear(hidden_units, self.input_dim))

    def forward(self, x):
        # X = X.view(X.shape[0], -1)
        # Xz = torch.cat([X, z], dim=-1)
        output = self.main(x)
        # used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))
        # output = output / used_sigmas
        return output


class NCSNv2SimpleS2(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.z_dim = config.model.z_dim
        self.register_buffer('sigmas', get_sigmas(config))
        self.input_dim=config.data.dim
        self.r = 8
        hidden_units= 1024
        self.main = nn.Sequential(
        nn.Linear(self.input_dim, hidden_units),
        nn.Softplus(),
        nn.Linear(hidden_units, hidden_units),
        nn.Softplus(),
        nn.Linear(hidden_units, self.input_dim * self.r),
        )

        self.alpha = nn.Sequential(
        nn.Linear(self.input_dim, hidden_units),
        nn.Softplus(),
        nn.Linear(hidden_units, hidden_units),
        nn.Softplus(),
        nn.Linear(hidden_units, self.input_dim),
        )

    def forward(self, x):
        # X = X.view(X.shape[0], -1)
        # Xz = torch.cat([X, z], dim=-1)
        output = self.main(x) #beta
        # used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))
        # output = output / used_sigmas
        alpha = torch.diag_embed(self.alpha(x))
        output = output.view(x.shape[0], self.input_dim, self.r)
        output = torch.bmm(output, output.permute(0, 2, 1))
        return (alpha + output).view(-1, self.input_dim**2)

class NCSNv2SimpleS3(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.z_dim = config.model.z_dim
        self.register_buffer('sigmas', get_sigmas(config))
        self.input_dim=config.data.dim
        self.r = 8
        hidden_units = 1024
        self.main = nn.Sequential(
        nn.Linear(self.input_dim, hidden_units),
        nn.Softplus(),
        nn.Linear(hidden_units, hidden_units),
        nn.Softplus(),
        nn.Linear(hidden_units, self.input_dim * self.input_dim * self.input_dim),
        )

        self.alpha = nn.Sequential(
        nn.Linear(self.input_dim, hidden_units),
        nn.Softplus(),
        nn.Linear(hidden_units, hidden_units),
        nn.Softplus(),
        nn.Linear(hidden_units, self.input_dim * self.input_dim),
        )

    def forward(self, x):
        # X = X.view(X.shape[0], -1)
        # Xz = torch.cat([X, z], dim=-1)
        output = self.main(x).view(-1, self.input_dim, self.input_dim, self.input_dim) #beta
        # used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))
        # output = output / used_sigmas
        alpha = torch.diag_embed(self.alpha(x).view(-1, self.input_dim, self.input_dim))
        # output = output.view(x.shape[0], self.input_dim, self.input_dim, self.r)
        # output = torch.bmm(output, output.permute(0, 2, 1))
        return (alpha + output).view(-1, self.input_dim**3)

class NCSNv2Deeper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.logit_transform = config.data.logit_transform
        self.rescaled = config.data.rescaled
        self.norm = get_normalization(config, conditional=False)
        self.ngf = ngf = config.model.ngf
        self.num_classes = config.model.num_classes
        self.act = act = get_act(config)
        self.register_buffer('sigmas', get_sigmas(config))
        self.config = config

        self.begin_conv = nn.Conv2d(config.data.channels, ngf, 3, stride=1, padding=1)
        self.normalizer = self.norm(ngf, self.num_classes)

        self.end_conv = nn.Conv2d(ngf, config.data.channels, 3, stride=1, padding=1)

        self.res1 = nn.ModuleList([
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm),
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res2 = nn.ModuleList([
            ResidualBlock(self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res3 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res4 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 4 * self.ngf, resample='down', act=act,
                          normalization=self.norm, dilation=2),
            ResidualBlock(4 * self.ngf, 4 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=2)]
        )

        self.res5 = nn.ModuleList([
            ResidualBlock(4 * self.ngf, 4 * self.ngf, resample='down', act=act,
                          normalization=self.norm, dilation=4),
            ResidualBlock(4 * self.ngf, 4 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=4)]
        )

        self.refine1 = RefineBlock([4 * self.ngf], 4 * self.ngf, act=act, start=True)
        self.refine2 = RefineBlock([4 * self.ngf, 4 * self.ngf], 2 * self.ngf, act=act)
        self.refine3 = RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, act=act)
        self.refine4 = RefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, act=act)
        self.refine5 = RefineBlock([self.ngf, self.ngf], self.ngf, act=act, end=True)

    def _compute_cond_module(self, module, x):
        for m in module:
            x = m(x)
        return x

    def forward(self, x, y):
        if not self.logit_transform and not self.rescaled:
            h = 2 * x - 1.
        else:
            h = x

        output = self.begin_conv(h)

        layer1 = self._compute_cond_module(self.res1, output)
        layer2 = self._compute_cond_module(self.res2, layer1)
        layer3 = self._compute_cond_module(self.res3, layer2)
        layer4 = self._compute_cond_module(self.res4, layer3)
        layer5 = self._compute_cond_module(self.res5, layer4)

        ref1 = self.refine1([layer5], layer5.shape[2:])
        ref2 = self.refine2([layer4, ref1], layer4.shape[2:])
        ref3 = self.refine3([layer3, ref2], layer3.shape[2:])
        ref4 = self.refine4([layer2, ref3], layer2.shape[2:])
        output = self.refine5([layer1, ref4], layer1.shape[2:])

        output = self.normalizer(output)
        output = self.act(output)
        output = self.end_conv(output)

        used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))

        output = output / used_sigmas

        return output

class NCSNv2Deepest(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.logit_transform = config.data.logit_transform
        self.rescaled = config.data.rescaled
        self.norm = get_normalization(config, conditional=False)
        self.ngf = ngf = config.model.ngf
        self.num_classes = config.model.num_classes
        self.act = act = get_act(config)
        self.register_buffer('sigmas', get_sigmas(config))
        self.config = config

        self.begin_conv = nn.Conv2d(config.data.channels, ngf, 3, stride=1, padding=1)
        self.normalizer = self.norm(ngf, self.num_classes)

        self.end_conv = nn.Conv2d(ngf, config.data.channels, 3, stride=1, padding=1)

        self.res1 = nn.ModuleList([
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm),
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res2 = nn.ModuleList([
            ResidualBlock(self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res3 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res31 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res4 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 4 * self.ngf, resample='down', act=act,
                          normalization=self.norm, dilation=2),
            ResidualBlock(4 * self.ngf, 4 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=2)]
        )

        self.res5 = nn.ModuleList([
            ResidualBlock(4 * self.ngf, 4 * self.ngf, resample='down', act=act,
                          normalization=self.norm, dilation=4),
            ResidualBlock(4 * self.ngf, 4 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=4)]
        )

        self.refine1 = RefineBlock([4 * self.ngf], 4 * self.ngf, act=act, start=True)
        self.refine2 = RefineBlock([4 * self.ngf, 4 * self.ngf], 2 * self.ngf, act=act)
        self.refine3 = RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, act=act)
        self.refine31 = RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, act=act)
        self.refine4 = RefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, act=act)
        self.refine5 = RefineBlock([self.ngf, self.ngf], self.ngf, act=act, end=True)

    def _compute_cond_module(self, module, x):
        for m in module:
            x = m(x)
        return x

    def forward(self, x, y):
        if not self.logit_transform and not self.rescaled:
            h = 2 * x - 1.
        else:
            h = x

        output = self.begin_conv(h)

        layer1 = self._compute_cond_module(self.res1, output)
        layer2 = self._compute_cond_module(self.res2, layer1)
        layer3 = self._compute_cond_module(self.res3, layer2)
        layer31 = self._compute_cond_module(self.res31, layer3)
        layer4 = self._compute_cond_module(self.res4, layer31)
        layer5 = self._compute_cond_module(self.res5, layer4)

        ref1 = self.refine1([layer5], layer5.shape[2:])
        ref2 = self.refine2([layer4, ref1], layer4.shape[2:])
        ref31 = self.refine31([layer31, ref2], layer31.shape[2:])
        ref3 = self.refine3([layer3, ref31], layer3.shape[2:])
        ref4 = self.refine4([layer2, ref3], layer2.shape[2:])
        output = self.refine5([layer1, ref4], layer1.shape[2:])

        output = self.normalizer(output)
        output = self.act(output)
        output = self.end_conv(output)

        used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))

        output = output / used_sigmas

        return output

class DSM_Generator_S2(nn.Module):
    def __init__(self, config):
        super(DSM_Generator_S2, self).__init__()
        self.input_dim=config.data.dim
        #nc = 1, nz = 100, ngf = 64
        ngf = 64#64
        nc = 1
        self.r = 8
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=self.input_dim, out_channels=ngf * 4, kernel_size=2, stride=1, padding=0, bias=False), #5
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 2 x 2
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # x- state size (ngf * 2) x4 x4
            nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            # x- state size. ngf x 6 x 6
            nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh(),
            # x- state size. 1 x 4 x 4
        )
        self.fc1 = nn.Linear(12*12, 100)
        self.fc2 = nn.Linear(100, self.input_dim*self.r)

        self.alpha = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=self.input_dim, out_channels=ngf * 4, kernel_size=2, stride=1, padding=0,
                               bias=False),  # 5
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 2 x 2
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # x- state size (ngf * 2) x4 x4
            nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # x- state size. ngf x 10 x 10
            nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh(),
        )
        self.alpha_fc1 = nn.Linear(10*10, 100)
        self.alpha_fc2 = nn.Linear(100, self.input_dim**2)

    def forward(self, input):
        batch_size = input.shape[0]
        d = input.shape[1]
        # if input.is_cuda and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        input = input.view(batch_size, d, 1, 1)
        beta = self.main(input)
        beta = beta.view(batch_size, -1)
        beta = F.relu(self.fc1(beta))
        beta = self.fc2(beta).view(batch_size, d, self.r)

        alpha = self.alpha(input).view(batch_size, -1)
        alpha = F.relu(self.alpha_fc1(alpha))
        alpha = self.alpha_fc2(alpha)
        # print("alpha,", alpha.shape)
        output = alpha.view(batch_size, -1) + torch.bmm(beta, beta.permute(0, 2, 1)).view(batch_size, -1)
        return output

class DSM_Generator_S1(nn.Module):
    def __init__(self, config):
        super(DSM_Generator_S1, self).__init__()
        self.input_dim=config.data.dim
        #nc = 1, nz = 100, ngf = 64
        ngf = 64#64
        nc = 1
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=self.input_dim, out_channels=ngf * 4, kernel_size=2, stride=1, padding=0, bias=False), #5
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 2 x 2
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # x- state size (ngf * 2) x4 x4
            nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=4, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            # nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.ReLU(True),
            # x- state size. ngf x 6 x 6
            nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh(),
            # x- state size. 1 x 4 x 4
        )
        self.fc1 = nn.Linear(14*14, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, input):
        batch_size = input.shape[0]
        d = input.shape[1]
        # if input.is_cuda and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        input = input.view(batch_size, d, 1, 1)
        output = self.main(input)
        output = output.view(batch_size, -1)
        output = F.relu(self.fc1(output))
        output = self.fc2(output)

        return output

class MLP_S2(nn.Module):
    def __init__(self, config):
        super(MLP_S2, self).__init__()
        self.input_dim=config.data.dim
        self.r = 7
        self.beta = nn.Sequential(
            nn.Linear(self.input_dim, 256),  # 256
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, self.input_dim * self.r)
        )

        self.alpha = nn.Sequential(
            nn.Linear(self.input_dim, 256),  # 256
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, self.input_dim * self.input_dim)
        )

    def forward(self, input):
        batch_size = input.shape[0]
        d = input.shape[1]
        # if input.is_cuda and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        # input = input.view(batch_size, d)
        beta = self.beta(input).view(batch_size, d, self.r)

        alpha = self.alpha(input)
        output = alpha.view(batch_size, -1) + torch.bmm(beta, beta.permute(0, 2, 1)).view(batch_size, -1)
        return output

class MLP_S1(nn.Module):
    def __init__(self, config):
        super(MLP_S1, self).__init__()
        self.input_dim=config.data.dim

        self.main = nn.Sequential(
            nn.Linear(self.input_dim, 256),  # 256
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, self.input_dim)
        )

    def forward(self, input):
        output = self.main(input)
        return output


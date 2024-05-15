import torch
import torch.nn as nn
import torch.nn.functional as F

#slice score matching
#for image
class Img_MLPScore(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.z_dim = config.model.z_dim
        self.input_dim=config.data.dim
        self.main = nn.Sequential(
            nn.Linear(self.input_dim, 256), #256
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, self.input_dim)
        )

    def forward(self, x):
        # X = X.view(X.shape[0], -1)
        # Xz = torch.cat([X, z], dim=-1)
        h = self.main(x)
        return h

#for image
class Img_Score(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nef = config.model.nef
        self.z_dim = config.model.z_dim
        self.channels = config.data.channels
        self.image_size = config.data.image_size
        self.dataset = config.data.dataset
        self.zfc = nn.Sequential(
            nn.Linear(self.z_dim, self.image_size ** 2),
            nn.Softplus()
        )
        self.main = nn.Sequential(
            nn.Conv2d(self.channels + 1, self.nef * 1, 5, stride=2, padding=2),
            nn.Softplus(),
            nn.Conv2d(self.nef * 1, self.nef * 2, 5, stride=2, padding=2),
            nn.Softplus(),
            nn.Conv2d(self.nef * 2, self.nef * 4, 5, stride=2, padding=2),
            nn.Softplus(),
            nn.Conv2d(self.nef * 4, self.nef * 8, 5, stride=2, padding=2),
            nn.Softplus()
        )
        self.flatten = nn.Sequential(
            nn.Linear((self.image_size // 2 ** 4) ** 2 * self.nef * 8, 512),
            nn.Softplus()
        )
        self.score = nn.Linear(512, self.z_dim)

    def forward(self, x):
        h = self.main(x)
        h = h.view(h.shape[0], -1)
        h = self.flatten(h)
        score = self.score(h)
        return score


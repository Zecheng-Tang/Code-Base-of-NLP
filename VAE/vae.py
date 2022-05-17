from cgitb import text
import os
from numpy import reshape
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, dim) -> None:
        super(VAE, self).__init__()
        
        self.encoder = nn.Sequential(nn.Linear(512, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 128))
        
        self.mu = nn.Linear(128, dim)
        self.logvar = nn.Linear(128, dim)

        self.latent_map = nn.Linear(dim, 128)

        self.decoder = nn.Sequential(nn.Linear(128, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 512))

    def encode(self, x):
        x = x.view(x.size(0), -1)
        en_res = self.encode(x)
        mu, logvar = self.mu(en_res), self.logvar(en_res)
        return mu, logvar

    def sample(self, mu, logvar):
        eps = torch.rand_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)
    
    def decode(self, z, x):
        latent_z = self.latent_map(z)
        out = self.decode(latent_z)
        reshape_out = torch.sigmoid(out).view(x.shape[0], 1, 28, 28)
        return reshape_out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        out = self.decode(z, x)
        return out


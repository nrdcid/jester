import torch.nn as nn
import torch
from dataclasses import dataclass

@dataclass
class Config:
    n_features: int
    n_hidden: int
    n_instances: int

class Autoencoder(nn.Module):
    def __init__(self, 
                 config,
                 feature_prob,
                 importance,
                 device):
        super(Autoencoder, self).__init__()
        self.config = config
        self.encoder = nn.Linear(config.n_features, config.n_hidden)
        self.decoder = nn.Linear(config.n_hidden, config.n_features)


    def forward(self, x):
        x = nn.ReLU(self.encoder(x))
        x = nn.Sigmoid(self.decoder(x))
        return x

def sparse_loss(x, r, h, lam=0.1):
    recon_loss = nn.functional.mse_loss(r, x)
    sparsity_penalty = lam * torch.sum(torch.abs(h))
    return recon_loss + sparsity_penalty
import torch.nn as nn
import torch

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, encoding_dim)
        self.decoder = nn.Linear(encoding_dim, input_dim)


    def forward(self, x):
        x = nn.ReLU(self.encoder(x))
        x = nn.Sigmoid(self.decoder())
        return x

class SparseAutoencoder(Autoencoder):
    def __init__(self, input_dim, encoding_dim):
        super(SparseAutoencoder, self).__init__(input_dim, encoding_dim)
        self.encoder = nn.Linear(input_dim, encoding_dim)
        self.decoder = nn.Linear(encoding_dim, input_dim)

    def forward(self, x):
        x = nn.ReLU(self.encoder(x))
        x = nn.Sigmoid(self.decoder(x))
        return x

def sparse_loss(x, r, h, lam=0.1):
    recon_loss = nn.functional.mse_loss(r, x)
    sparsity_penalty = lam * torch.sum(torch.abs(h))
    return recon_loss + sparsity_penalty
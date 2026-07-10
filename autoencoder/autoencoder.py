import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(28 * 28, encoding_dim)
        self.decoder = nn.Linear(encoding_dim, 28 * 28)


    def forward(self, x):
        x = nn.ReLU(self.encoder(x))
        x = nn.Sigmoid(self.decoder())
        return x

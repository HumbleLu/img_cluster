import torch
import torch.nn as nn

import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_size = 1024):
        super().__init__()
        self.encoder = nn.Sequential(
                nn.Linear(input_size, 500), 
                nn.ReLU(),
                nn.Linear(500, 500), 
                nn.ReLU(),
                nn.Linear(500, 2000), 
                nn.ReLU()
        )

        self.decoder = nn.Sequential(
                nn.Linear(10, 2000), 
                nn.ReLU(),
                nn.Linear(2000, 500), 
                nn.ReLU(),
                nn.Linear(500, 500), 
                nn.ReLU(),
                nn.Linear(500, input_size)
        )

        self.fc_2_mu = nn.Linear(2000, 10)
        self.fc_2_ln_sigma = nn.Linear(2000, 10)

    def reparams(self, mu, sigma):
        epsilon = torch.randn(mu.size())
        z = mu + epsilon * sigma
        return z

    def encode(self, x):
        h = self.encoder(x)
        mu, ln_sigma = self.fc_2_mu(h), self.fc_2_ln_sigma(h)
        sigma = ln_sigma.exp()
        z = self.reparams(mu, sigma)
        return z, mu, sigma

    def decode(self, z):
        recons_x = self.decoder(z)
        return recons_x

    def forward(self, x):
        z, mu, sigma = self.encode(x)
        recons_x = self.decode(z)
        return recons_x, z, mu, sigma

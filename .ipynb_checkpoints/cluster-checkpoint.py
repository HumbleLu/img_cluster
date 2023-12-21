import torch
import torch.nn as nn
from torch.nn import Parameter

class Cluster(nn.Module):
    def __init__(self, init_centroids, alpha = 1):
        super().__init__()
        self.centroids = Parameter(init_centroids)
        self.centroids.requires_grad = True
        self.alpha = alpha

    def compute_q(self, z):
        sse = (z.unsqueeze(1) - self.centroids) ** 2
        sse = torch.sum(sse, dim = 2) # sum over embedding dimension
        q = (1 + sse / self.alpha) ** (-(self.alpha + 1)/2)
        q = (q.t() / torch.sum(q, dim = 1)).t() # sum over centroids
        return q

    def compute_p(self, q):
        sf = (q**2) / torch.sum(q, dim = 0) # sim over inputs
        p = (sf.t() / torch.sum(sf, dim = 1)).t() # sum over centroids
        return p
    
    def forward(self, z):
        q = self.compute_q(z)
        p = self.compute_p(q)
        return p, q
import torch.nn as nn

class DELIUS(nn.Module):
    def __init__(self, feature_model, vae_model, feature_dim = 1024, kernel_size = (7, 7)):
        super().__init__()
        self.feature_model = feature_model
        self.vae_model = vae_model
        self.feature_dim = feature_dim
        self.global_pool = nn.AvgPool2d(kernel_size = kernel_size)
    
    def forward(self, x):
        h = self.feature_model(x)
        h = self.global_pool(h).view([-1, self.feature_dim])
        recons_h, z, _, _ = self.vae_model(h)
        return recons_h, h, z

import sys, os
# path to the models I made
sys.path.insert(0, '/projappl/project_2005600/wallpaper/img_clustering')
os.environ['TORCH_HOME'] = '/scratch/project_2005600'

cluster_num = int(sys.argv[1])
batch_size = int(sys.argv[2])
epochs = int(sys.argv[3])
learning_rate = float(sys.argv[4])

print(f'clusters: {sys.argv[1]}')
print(f'batch size: {sys.argv[2]}')
print(f'epochs: {sys.argv[3]}')
print(f'learning rate: {sys.argv[4]}')

# required torch packages
import torch
import torch.nn as nn
from torch.nn import functional as F

# import models I made
from vae import VAE
from delius import DELIUS
from cluster import Cluster
from wallpaper_workshop_dataset import WallpaperWorkshopDataset

# set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data processing
from torchvision import transforms
csv_file = '/scratch/project_2005600/wallpaper_jepg_rgb.csv'
root_dir = '/scratch/project_2005600/wallpaper_workshop'
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias = True),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

ww_dataset = WallpaperWorkshopDataset(csv_file, root_dir, img_transform)

# data loader
from torch.utils.data import DataLoader
dataloader = DataLoader(ww_dataset, batch_size = 32, shuffle = True)

# feature extraction model
densenet_model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True).to(device)

vae_model = VAE(1024).to(device)
feature_model = DELIUS(densenet_model.features, vae_model)

# optional: freeze the feature extraction (from a pre-trained model) layers
for params in feature_model.feature_model:
    params.requires_grad = False

feature_model.to(device)

# clustering model
centroids = torch.randn([cluster_num, 10], requires_grad = True)
centroids.to(torch.float32)
cluster_model = Cluster(centroids)
cluster_model.to(device)

# training utilities/ parameters
optimizer = torch.optim.Adam(
    list(feature_model.parameters()) + list(cluster_model.parameters()),
    lr = learning_rate)

# training
n_steps = len(dataloader)
for epoch in range(epochs):
    for i, x in enumerate(dataloader):
        # feature extraction
        # feed forward
        image = x['image'].to(device)
        recons_h, h, z, mu, sigma= feature_model(image)
        
        recons_h = recons_h.to(device)
        h = h.to(device)
        z = z.to(device)
        
        feature_loss = F.mse_loss(recons_h, h)

        # cluster
        # feed forward
        p, q = cluster_model(z)

        p = p.to(device)
        q = q.to(device)
        
        cluster_loss = F.kl_div(torch.log(p), torch.log(q), log_target = True)
        # cluster_loss.backward()
        
        # compute loss and gradients
        total_loss =  feature_loss + cluster_loss
        total_loss.backward()

        # update parameters
        optimizer.step()

        # print status
        if (i + 1) % 10 == 0:
            print(f'epoch: [{epoch+1}/{epochs}]')
            print(f'[{i+1}/{n_steps}]')
    
            print(f'total loss: {total_loss.item()}')
            print(f'feature loss: {feature_loss.item()}')
            print(f'cluster loss: {cluster_loss.item()}')
            
        optimizer.zero_grad()
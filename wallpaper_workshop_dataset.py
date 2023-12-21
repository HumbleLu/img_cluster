import os, sys
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class WallpaperWorkshopDataset(Dataset):
    """Wallpaper workshop dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with popularity metrics.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.pop_metrics = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.pop_metrics)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir,
                                str(self.pop_metrics.iloc[idx, 0]))
        
        image = plt.imread(img_path)

        if self.transform:
            image = self.transform(image)
        
        sample = {'image': image, 
                  'num_visitor': self.pop_metrics.iloc[idx, 3], 
                  'num_subscriber': self.pop_metrics.iloc[idx, 4], 
                  'num_favorite': self.pop_metrics.iloc[idx, 5]}

        return sample
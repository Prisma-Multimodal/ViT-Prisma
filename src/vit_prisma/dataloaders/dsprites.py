import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
import os


class DSpritesDataset(Dataset):
    def __init__(self, data_path= '/network/scratch/s/firstname.joseph/datasets/dsprites.npz'):
        # Load the dataset
        data = np.load(data_path, allow_pickle=True, encoding='latin1')

        self.images = data['imgs']
        self.labels =  data['latents_values'][:,1]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Convert image from numpy array to PyTorch tensor and add channel dimension
        image = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx]-1, dtype=torch.int64)
        return image, label
    

def train_test_dataset(dataset, test_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=test_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['test'] = Subset(dataset, val_idx)
    return datasets


# Example to load the data with train and test split
## dsprites_dataset = DSpritesDataset(data_path)
## dsprites_datasets = train_test_dataset(dsprites_dataset, test_split=0.25)

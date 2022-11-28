import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class LorenzDataset(Dataset):

    def __init__(self, fpath):
        self.x = torch.from_numpy(np.load(fpath + "x_train.npy"))
        self.x_dot = torch.from_numpy(np.load(fpath + "x_dot.npy"))

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.x_dot[idx]  

class RosslerDataset(Dataset):

    def __init__(self, fpath):
        self.x = torch.from_numpy(np.load(fpath + "x_train.npy"))
        self.x_dot = torch.from_numpy(np.load(fpath + "x_dot.npy"))

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.x_dot[idx]  

class PupilDataset(Dataset):

    def __init__(self, datapath="../../data/pupil1.npy", norm=True, scale=1.0, amount=None):
        pupil = np.load(datapath)
        if amount is not None:
            pupil = pupil[0:amount]
        pupil = pupil[700:]
        if norm:
            pupil = (pupil - np.mean(pupil)) / np.std(pupil)
        pupil = pupil * scale
        d_pupil = np.diff(pupil, axis=0) / 0.05
        pupil = pupil[:-1]
        self.x = torch.from_numpy(pupil)
        self.x_dot = torch.from_numpy(d_pupil)

    def __len__(self):
        return self.x.size(0)
    
    def __getitem__(self, idx):
        return self.x[idx], self.x_dot[idx]      
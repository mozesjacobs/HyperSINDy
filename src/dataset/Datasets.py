import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from src.utils.model_utils import library_size, sindy_library

class SyntheticDataset(Dataset):

    def __init__(self, args, hyperparams, fpath):
        self.x = torch.from_numpy(np.load(fpath + "x_train.npy"))
        #self.x_dot = torch.from_numpy(np.load(fpath + "x_dot.npy"))
        self.x_dot = self.fourth_order_diff(self.x, args.dt)
        self.x_lib = sindy_library(
            self.x, args.poly_order,
            include_constant=args.include_constant,
            include_sine=args.include_sine)

        # If training on ESDINy, build ensemble
        if args.model == "ESINDy":
            # timesteps x num_ensemble x dim
            num_ensemble = hyperparams.num_ensemble
            idx = torch.randint(0, x.size(0), (num_ensemble, x.size(0)))
            self.x = self.build_ensemble(self.x, idx)
            self.x_dot = self.build_ensemble(self.x_dot, idx)
            self.x_lib = self.build_ensemble(self.x_lib, idx)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.x_lib[idx], self.x_dot[idx]

    def fourth_order_diff(self, x, dt):
        dx = torch.zeros(x.size())
        dx[0] = (-11.0 / 6) * x[0] + 3 * x[1] - 1.5 * x[2] + x[3] / 3
        dx[1] = (-11.0 / 6) * x[1] + 3 * x[2] - 1.5 * x[3] + x[4] / 3
        dx[2:-2] = (-1.0 / 12) * x[4:] + (2.0 / 3) * x[3:-1] - (2.0 / 3) * x[1:-3] + (1.0 / 12) * x[:-4]
        dx[-2] = (11.0 / 6) * x[-2] - 3.0 * x[-3] + 1.5 * x[-4] - x[-5] / 3.0
        dx[-1] = (11.0 / 6) * x[-1] - 3.0 * x[-2] + 1.5 * x[-3] - x[-4] / 3.0
        return dx / dt 

    def build_ensemble(self, data, idx):
        return torch.transpose(data[idx], 0, 1)
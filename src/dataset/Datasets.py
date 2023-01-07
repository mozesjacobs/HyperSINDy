import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from src.utils.model_utils import library_size, sindy_library

class SyntheticDataset(Dataset):
    """A dataset to run experiments with.

    Creates a dataset of torch tensors. Tensors must be loaded from numpy
    array files.

    Attributes:
        self.x: The data as a torch.Tensor of shape (timesteps x z_dim) for
            HyperSINDy or SINDy or shape (timesteps x num_ensemble x z_dim)
            for ESINDy. In the case of ESINDy, self.x is created through
            bootstrapping.
        self.x_dot: The derivative with respect to time of self.x, as a
            torch.Tensor of shape (timesteps x z_dim) for HyperSINDy or
            SINDy or shape (timesteps x num_ensemble x z_dim) for ESINDy.
            In the case of ESINDy, self.x_dot is created through bootstrapping.
            The derivative is calculated using fourth order differentiation.
        self.x_lib: The sindy_library form of self.x, as a torch.Tensor of
            of shape (timesteps x library_dim) for HyperSINDy or SINDy or
            shape (timesteps x num_ensemble x library_dim) for ESINDy. In the
            case of ESINDy, self.x_lib is created through boostrapping.
        self.x_true: For HyperSINDy or SINDy, a reference to self.x. For ESINDy,
            it is self.x before bootstrapping was performed.
    """

    def __init__(self, args, hyperparams, fpath):
        """Initializes the SyntheticDataset.

        Initializes the SyntheticDataset using the given arguments and
        hyperparameters. In addition, loads data from the given file path.

        Args:
            args: The argparser object return by parse_args() in the file
                cmd_line.py.
            hyperparams: The argparser object returned by parse_hyperparams() in 
                the file cmd_line.py
            fpath: The path to the data file, excluding the file name ("x_train.numpy").

        Returns:
            A SyntheticDataset.
        """
        self.x = torch.from_numpy(np.load(fpath + "x_train.npy"))
        #self.x_dot = torch.from_numpy(np.load(fpath + "x_dot.npy"))
        self.x_dot = self.fourth_order_diff(self.x, args.dt)
        self.x_lib = sindy_library(
            self.x, args.poly_order,
            include_constant=args.include_constant,
            include_sine=args.include_sine)
        self.x_true = self.x

        # If training on ESINDy, build ensemble
        if args.model == "ESINDy":
            # timesteps x num_ensemble x dim
            num_ensemble = hyperparams.num_ensemble
            idx = torch.randint(0, self.x.size(0), (num_ensemble, self.x.size(0)))
            self.x = self.build_ensemble(self.x, idx)
            self.x_dot = self.build_ensemble(self.x_dot, idx)
            self.x_lib = self.build_ensemble(self.x_lib, idx)
            

    def __len__(self):
        """The length of the dataset.

        Gets the length of the dataset (in timesteps).

        Args:
            None

        Returns:
            The length of the dataset along dimension 0.
        """
        return len(self.x)
    
    def __getitem__(self, idx):
        """Gets the item.

        Gets the item at the current index.

        Args:
            idx: The integer index to access the data.

        Returns:
            A tuple of (tensor_a, tensor_b, tensor_c), where tensor_a is
            self.x[idx], tensor_b is self.x_lib[idx], and tensor_c is
            self.x_dot[idx].
        """
        return self.x[idx], self.x_lib[idx], self.x_dot[idx]

    def fourth_order_diff(self, x, dt):
        """Gets the derivatives of the data.

        Gets the derivative of x with respect to time using fourth order
        differentiation.

        Args:
            x: The data (torch.Tensor of shape (timesteps x z_dim)) to
                differentiate.
            dt: The amount of time between two adjacent data points (i.e.,
                the time between x[0] and x[1], or x[1] and x[2]).

        Returns:
            A torch.tensor of the derivatives of x.
        """
        dx = torch.zeros(x.size())
        dx[0] = (-11.0 / 6) * x[0] + 3 * x[1] - 1.5 * x[2] + x[3] / 3
        dx[1] = (-11.0 / 6) * x[1] + 3 * x[2] - 1.5 * x[3] + x[4] / 3
        dx[2:-2] = (-1.0 / 12) * x[4:] + (2.0 / 3) * x[3:-1] - (2.0 / 3) * x[1:-3] + (1.0 / 12) * x[:-4]
        dx[-2] = (11.0 / 6) * x[-2] - 3.0 * x[-3] + 1.5 * x[-4] - x[-5] / 3.0
        dx[-1] = (11.0 / 6) * x[-1] - 3.0 * x[-2] + 1.5 * x[-3] - x[-4] / 3.0
        return dx / dt 

    def build_ensemble(self, data, idx):
        """Builds an ensemble.

        Builds an ensemble of the data using the given boostrap indices.

        Args:
            data: The data to boostrap. Should be a torch.Tensor of shape
                (timesteps x -1).
            idx: A torch.Tensor denoting the bootstrap indices. Should be of shape
                (num_ensemble, timesteps).
        
        Returns:
            A torch.Tensor of the boostrapped data. The tensor will have shape
            (timesteps x num_ensemble x -1).
        """
        return torch.transpose(data[idx], 0, 1)
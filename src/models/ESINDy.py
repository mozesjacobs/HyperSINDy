import torch
import torch.nn as nn
from src.models.HyperNet import HyperNet
from src.utils.model_utils import library_size, sindy_library


class Net(nn.Module):
    def __init__(self, args, hyperparams):
        super(Net, self).__init__()
        
        self.z_dim = args.z_dim
        self.poly_order = args.poly_order
        self.include_constant = args.include_constant
        self.include_sine = args.include_sine
        self.statistic_batch_size = args.statistic_batch_size

        self.hypernet_hidden_dim = hyperparams.hidden_dim
        self.num_ensemble = hyperparams.num_ensemble
        self.prior = hyperparams.prior

        self.library_dim = library_size(self.z_dim, self.poly_order,
            include_constant=self.include_constant, use_sine=self.include_sine)
        
        self.sindy_coefs = nn.Parameter(
            #torch.normal(0, 0.1, (self.num_ensemble, self.library_dim, self.z_dim)),
            torch.ones(self.num_ensemble, self.library_dim, self.z_dim),
            requires_grad=True)

        self.threshold_mask = nn.Parameter(
            torch.ones(self.num_ensemble, self.library_dim, self.z_dim),
            requires_grad=False)
    
        
    def forward(self, x, x_lib=None, device=0):
        x = x.type(torch.FloatTensor).to(device)
        if x_lib is None:
            x_lib = self.make_library(x)
        else:
            x_lib = x_lib.type(torch.FloatTensor).to(device)

        coeffs = self.get_masked_coefficients(batch_size=x.size(0), device=device)
        return self.dx(x_lib, coeffs), coeffs
    
    def dx(self, library, coefs):
        return torch.matmul(library.unsqueeze(2), coefs).squeeze(2)

    def get_masked_coefficients(self, n=None, batch_size=None, device=0):
        return self.sindy_coefs * self.threshold_mask

    def update_threshold_mask(self, threshold, device):
        self.threshold_mask[torch.abs(self.sindy_coefs) < threshold] = 0

    def make_library(self, x):
        return sindy_library(x, self.poly_order,
                             include_constant=self.include_constant,
                             include_sine=self.include_sine)
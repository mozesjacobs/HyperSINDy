import torch
import torch.nn as nn
from src.utils.model_utils import library_size, sindy_library


class Net(nn.Module):
    def __init__(self, args, hyperparams):
        super(Net, self).__init__()

        self.z_dim = args.z_dim
        self.poly_order = args.poly_order
        self.include_constant = args.include_constant
        self.include_sine = args.include_sine
        self.statistic_batch_size = args.statistic_batch_size
        self.prior = hyperparams.prior

        self.library_dim = library_size(self.z_dim, self.poly_order,
            include_constant=self.include_constant, use_sine=self.include_sine)
        
        self.threshold_mask = nn.Parameter(torch.ones(self.library_dim, self.z_dim), requires_grad=False)
        self.sindy_coefficients = nn.Parameter(torch.normal(0, 0.1, (self.library_dim, self.z_dim)), requires_grad=True)
        
    def forward(self, x, x_lib=None, device=0):
        x = x.type(torch.FloatTensor).to(device)
        if x_lib is None:
            x_lib = sindy_library(x, self.poly_order,
                                  include_constant=self.include_constant,
                                  include_sine=self.include_sine)
        else:
            x_lib = x_lib.type(torch.FloatTensor).to(device)
        coefs = self.get_masked_coefficients()
        return self.dz(x, coefs), coefs

    def get_masked_coefficients(self):
        return self.sindy_coefficients * self.threshold_mask
    
    def dz(self, x, coefs):
        library = sindy_library(x, self.poly_order, include_constant=self.include_constant, include_sine=self.include_sine)
        theta = torch.matmul(library, coefs).squeeze(1)
        return theta

    def update_threshold_mask(self, threshold):
        self.threshold_mask[torch.abs(self.sindy_coefficients) < threshold] = 0
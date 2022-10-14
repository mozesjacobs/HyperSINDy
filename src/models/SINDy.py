import torch
import torch.nn as nn
from src.utils.model_utils import library_size, sindy_library


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        self.z_dim = args.z_dim
        self.poly_order = args.poly_order
        self.noise_dim = args.noise_dim
        self.include_constant = args.include_constant
        self.include_sine = args.include_sine
        self.statistic_batch_size = args.statistic_batch_size

        self.library_dim = library_size(self.z_dim, self.poly_order,
            include_constant=self.include_constant, use_sine=self.include_sine)
        
        self.threshold_mask = nn.Parameter(torch.ones(self.library_dim, self.z_dim), requires_grad=False)
        self.sindy_coefficients = nn.Parameter(torch.normal(0, 0.1, (self.library_dim, self.z_dim)), requires_grad=True)
        
    def forward(self, x, device):
        x = x.type(torch.FloatTensor).to(device)
        return self.dz(x, self.sindy_coefficients), self.sindy_coefficients
    
    def dz(self, x, sindy_coeffs):
        library = sindy_library(x, self.poly_order, include_constant=self.include_constant, include_sine=self.include_sine)
        masked_coefficients = sindy_coeffs * self.threshold_mask
        theta = torch.matmul(library, masked_coefficients).squeeze(1)
        return theta
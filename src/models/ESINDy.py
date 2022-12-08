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
        self.noise_dim = hyperparams.noise_dim

        self.library_dim = library_size(self.z_dim, self.poly_order,
            include_constant=self.include_constant, use_sine=self.include_sine)
        self.hypernet = HyperNet(self.noise_dim, (self.library_dim, self.z_dim),
            [self.hypernet_hidden_dim for _ in range(4)])

        self.threshold_mask = nn.Parameter(torch.ones(self.library_dim, self.z_dim),
            requires_grad=False)
    
        
    def forward(self, x, x_lib=None, device=0):
        x = x.type(torch.FloatTensor).to(device)
        if x_lib is None:
            x_lib = sindy_library(x, self.poly_order,
                                  include_constant=self.include_constant,
                                  include_sine=self.include_sine)
        else:
            x_lib = x_lib.type(torch.FloatTensor).to(device)

        coeffs = self.get_masked_coefficients(batch_size=x.size(0), device=device)
        return self.dx(x_lib, coeffs), coeffs
    
    def dx(self, library, coefs):
        return torch.bmm(library.unsqueeze(1), coefs).squeeze(1)

    def sample_coeffs(self, n=None, batch_size=None, device=0):
        if batch_size is None:
            batch_size = self.statistic_batch_size
        if n is None:
            n = torch.randn([batch_size, self.noise_dim], device=device)
        return self.hypernet(n)

    def get_masked_coefficients(self, n=None, batch_size=None, device=0):
        return self.sample_coeffs(n, batch_size, device) * self.threshold_mask

    def update_threshold_mask(self, threshold, device):
        coefs = torch.mean(self.get_masked_coefficients(device=device), dim=0)
        self.threshold_mask[torch.abs(coefs) < threshold] = 0
    
    # KL function taken from:
    # https://github.com/pawni/BayesByHypernet_Pytorch/blob/master/model.py
    def kl(self, sindy_coeffs):
        num_samples = sindy_coeffs.size(0)
        masked_coeffs = sindy_coeffs.reshape(num_samples, -1) # 250 x 60
        gen_weights = masked_coeffs.transpose(1, 0) # 60 x 250
        prior_samples = torch.randn_like(gen_weights)
        eye = torch.eye(num_samples, device=gen_weights.device) # 250 x 250
        wp_distances = (prior_samples.unsqueeze(2) - gen_weights.unsqueeze(1)) ** 2  # 60 x 250 x 250
        ww_distances = (gen_weights.unsqueeze(2) - gen_weights.unsqueeze(1)) ** 2    # 60 x 250 x 250

        # zero out indices that were thresholded so kl isn't calculated for them
        #wp_distances = wp_distances * self.threshold_mask
        #ww_distances = ww_distances * self.threshold_mask
        
        wp_distances = torch.sqrt(torch.sum(wp_distances, 0) + 1e-8) # 250 x 250
        wp_dist = torch.min(wp_distances, 0)[0] # 250
        ww_distances = torch.sqrt(torch.sum(ww_distances, 0) + 1e-8) + eye * 1e10 # 250 x 250
        ww_dist = torch.min(ww_distances, 0)[0] # 250

        # mean over samples
        kl = torch.mean(torch.log(wp_dist / (ww_dist + 1e-8) + 1e-8))
        kl *= gen_weights.shape[0]
        kl += torch.log(torch.tensor(float(num_samples) / (num_samples - 1)))
        return kl
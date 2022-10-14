import torch
import torch.nn as nn
from src.models.HyperNet import HyperNet
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
        self.hypernet = HyperNet(self.noise_dim, (1, self.z_dim),
            [args.hidden_dim for _ in range(4)])
        self.threshold_mask = nn.Parameter(torch.ones(self.library_dim, self.z_dim),
            requires_grad=False)
        
        # init sindy coefficients
        the_output = self.hypernet(torch.randn([self.statistic_batch_size, self.noise_dim])).detach().clone()
        the_mean, the_std = torch.mean(the_output), torch.std(the_output)
        self.sindy_coefficients = nn.Parameter(torch.normal(the_mean, the_std, (self.library_dim - 1, self.z_dim)), requires_grad=True)
        
    def forward(self, x, device):
        x = x.type(torch.FloatTensor).to(device)
        n = torch.randn([x.size(0), self.noise_dim], device=device)
        sindy_coeffs = self.sample_transition(n=n, device=device)
        sindy_coeffs2 = self.sindy_coefficients.unsqueeze(0).repeat(sindy_coeffs.size(0), 1, 1)
        sindy_coeffs_full = torch.cat((sindy_coeffs, sindy_coeffs2), dim=1)
        return self.dz(x, sindy_coeffs_full), sindy_coeffs_full, sindy_coeffs
    
    def sample_transition(self, n=None, batch_size=1, device='cpu'):
        return self.hypernet(n, batch_size, device=device)
    
    def dz(self, x, sindy_coeffs):
        library = sindy_library(x, self.poly_order, include_constant=self.include_constant, include_sine=self.include_sine)
        masked_coefficients = sindy_coeffs * self.threshold_mask
        library = library.unsqueeze(1)
        theta = torch.bmm(library, masked_coefficients).squeeze(1)
        return theta

    def sindy_coeffs_full(self, batch_size=500, device=0):
        sindy_coeffs = self.sample_transition(batch_size=batch_size, device=device)
        sindy_coeffs2 = self.sindy_coefficients.unsqueeze(0).repeat(sindy_coeffs.size(0), 1, 1)
        return torch.cat((sindy_coeffs, sindy_coeffs2), dim=1)

    def get_masked_coefficients(self, batch_size=500, device=0):
        return self.sindy_coeffs_full(batch_size, device) * self.threshold_mask
    
    def kl(self, sindy_coeffs, num_samples=5, full_kernel=True):
        num_samples = sindy_coeffs.size(0)
    
        masked_coeffs = sindy_coeffs.reshape(num_samples, -1) # 250 x 60
        gen_weights = masked_coeffs.transpose(1, 0) # 60 x 250
        prior_samples = torch.randn_like(gen_weights)
        eye = torch.eye(num_samples, device=gen_weights.device) # 250 x 250
        wp_distances = (prior_samples.unsqueeze(2) - gen_weights.unsqueeze(1)) ** 2  # 60 x 250 x 250
        ww_distances = (gen_weights.unsqueeze(2) - gen_weights.unsqueeze(1)) ** 2    # 60 x 250 x 250
        
        # anything thresholded out to 0 should have 0 loss (so gradient doesn't flow)
        wp_distances = wp_distances * self.threshold_mask[0].reshape(-1, 1, 1)
        ww_distances = ww_distances * self.threshold_mask[0].reshape(-1, 1 ,1)
        
        wp_distances = torch.sqrt(torch.sum(wp_distances, 0) + 1e-8) # 250 x 250
        wp_dist = torch.min(wp_distances, 0)[0] # 250
        ww_distances = torch.sqrt(torch.sum(ww_distances, 0) + 1e-8) + eye * 1e10 # 250 x 250
        ww_dist = torch.min(ww_distances, 0)[0] # 250

        # mean over samples
        kl = torch.mean(torch.log(wp_dist / (ww_dist + 1e-8) + 1e-8))
        kl *= gen_weights.shape[0]
        kl += torch.log(torch.tensor(float(num_samples) / (num_samples - 1)))
        return kl
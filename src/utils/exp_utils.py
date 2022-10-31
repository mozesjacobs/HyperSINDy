import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from src.utils.other import *
from src.utils.plotting import *
from src.utils.model_utils import equation_sindy_library, sindy_coeffs_stats, get_equation


# returns: batch_size x ts x z_dim
def sample_trajectory(net, device, init_cond, batch_size=10, dt=1e-2, ts=5000):
    zc = torch.from_numpy(init_cond).type(torch.FloatTensor).to(device)
    zc = torch.stack([zc for _ in range(batch_size)], dim=0)
    zs = []
    for i in range(ts):
        zc = zc + net(zc, device)[0] * dt
        zs.append(zc)
    zs = torch.stack(zs, dim=0)
    zs = torch.transpose(zs, 0, 1)
    return zs.detach().cpu().numpy()

def update_equation_list(equations, library, coefs, starts, z_dim):
    for i in range(z_dim):
        equations.append(get_equation(library, coefs[:,i], starts[i]))

def get_equations(net, model_type, device, batch_size, z_dim, poly_order, include_constant, include_sine, include_mult_sine):
    starts = ["X' = ", "Y' = ", "Z' = "]
    library = equation_sindy_library(z_dim, poly_order,
                                         include_constant=include_constant,
                                         include_sine=include_sine,
                                         include_mult_sine=include_mult_sine)
    equations = []
    if model_type == "HyperSINDy1":
        coefs = net.get_masked_coefficients(batch_size=batch_size, device=device)
        mean_coefs, std_coefs = sindy_coeffs_stats(coefs)
        equations.append("MEAN")
        update_equation_list(equations, library, mean_coefs.detach().cpu().numpy(), starts, z_dim)
        equations.append("STD")
        update_equation_list(equations, library, std_coefs.detach().cpu().numpy(), starts, z_dim)
    elif model_type == "HyperSINDy2":
        coefs = net.get_masked_coefficients(batch_size=batch_size, device=device)
        mean_coefs, std_coefs = sindy_coeffs_stats(coefs)
        equations.append("MEAN")
        update_equation_list(equations, library, mean_coefs.detach().cpu().numpy(), starts, z_dim)
        equations.append("STD")
        update_equation_list(equations, library, std_coefs.detach().cpu().numpy(), starts, z_dim)
    elif model_type == "HyperSINDy3" or model_type == "HyperSINDy22":
        sindy_coefs = net.sindy_coefficients * net.threshold_mask
        noise_coefs = net.sample_transition(batch_size=batch_size, device=device) * net.threshold_mask_noise
        mean_true_coefs = torch.mean(sindy_coefs + noise_coefs, dim=0)
        equations.append("DETERMINISTIC + NOISE")
        update_equation_list(equations, library, mean_true_coefs.detach().cpu().numpy(), starts, z_dim)
        equations.append("DETERMINISTIC")
        update_equation_list(equations, library, sindy_coefs.detach().cpu().numpy(), starts, z_dim)
        mean_coefs, std_coefs = sindy_coeffs_stats(noise_coefs)
        equations.append("NOISE MEAN")
        update_equation_list(equations, library, mean_coefs.detach().cpu().numpy(), starts, z_dim)
        equations.append("NOISE STD")
        update_equation_list(equations, library, std_coefs.detach().cpu().numpy(), starts, z_dim)
    elif model_type == "SINDy":
        sindy_coefs = net.sindy_coefficients * net.threshold_mask
        equations.append("SINDy")
        update_equation_list(equations, library, sindy_coefs.detach().cpu().numpy(), starts, z_dim)
    return equations























def plot_weight_distribution(fpath, coeffs):
    coeffs = coeffs.detach().cpu().numpy()
    sns.set()
    fig, axes = plt.subplots(1, coeffs.shape[1], figsize=(10, 5))
    for i in range(coeffs.shape[1]):
        sns.histplot(coeffs[:,i], ax=axes[i], kde=True)
    plt.savefig(fpath)
    plt.close()
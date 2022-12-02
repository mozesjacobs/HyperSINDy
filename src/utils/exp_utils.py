import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from src.utils.other import *
from src.utils.plotting import *
from src.utils.model_utils import equation_sindy_library


# returns: batch_size x ts x z_dim
def sample_trajectory(net, device, x0, batch_size=10, dt=1e-2, ts=5000):
    zc = torch.from_numpy(x0).type(torch.FloatTensor).to(device)
    zc = torch.stack([zc for _ in range(batch_size)], dim=0)
    zs = []
    for i in range(ts):
        zc = zc + net(zc, device=device)[0] * dt
        zs.append(zc)
    zs = torch.stack(zs, dim=0)
    zs = torch.transpose(zs, 0, 1)
    return zs.detach().cpu().numpy()

def build_equation(lib, coef, eq):
    for i in range(len(coef)):
        if coef[i] != 0:
            if i == len(coef) - 1:
                eq += str(coef[i]) + lib[i]
            else:
                eq += str(coef[i]) + lib[i] + ' + '
    if eq[-2] == '+':
        eq = eq[:-3]
    return eq

def update_equation_list(equations, library, coefs, starts, z_dim):
    for i in range(z_dim):
        equations.append(build_equation(library, coefs[:,i], starts[i]))

def get_equations(net, model_type, device, z_dim, poly_order, include_constant, include_sine):
    starts = ["X' = ", "Y' = ", "Z' = "]
    library = equation_sindy_library(z_dim, poly_order, include_constant=include_constant, include_sine=include_sine)
    equations = []
    if model_type == "HyperSINDy":
        # drift
        sindy_coefs = net.get_masked_coefficients().detach().cpu().numpy()
        equations.append("Drift")
        update_equation_list(equations, library, sindy_coefs, starts, z_dim)
        # diffusion
        diff_term = net.sample_diffusion(device=device)
        mean_diff, std_diff = sindy_coeffs_stats(diff_term)
        equations.append("Drift Mean")
        update_equation_list(equations, library, mean_diff, starts, z_dim)
        equations.append("Drift STD")
        update_equation_list(equations, library, std_diff, starts, z_dim)
    elif model_type == "SINDy":
        sindy_coefs = net.sindy_coefficients * net.threshold_mask
        equations.append("SINDy")
        update_equation_list(equations, library, sindy_coefs, starts, z_dim)
    return equations

def plot_weight_distribution(fpath, coeffs):
    coeffs = coeffs.detach().cpu().numpy()
    sns.set()
    fig, axes = plt.subplots(1, coeffs.shape[1], figsize=(10, 5))
    for i in range(coeffs.shape[1]):
        sns.histplot(coeffs[:,i], ax=axes[i], kde=True)
    plt.savefig(fpath)
    plt.close()

def sindy_coeffs_stats(sindy_coeffs):
    coefs = sindy_coeffs.detach().cpu().numpy()
    return np.mean(coefs, axis=0), np.std(coefs, axis=0)
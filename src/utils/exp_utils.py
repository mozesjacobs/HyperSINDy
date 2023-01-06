import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from src.utils.other import *
from src.utils.plotting import *
from src.utils.model_utils import equation_sindy_library

def sample_trajectory(net, device, x0, batch_size=10, dt=1e-2, ts=5000):
    """Generates trajectories.

    Generates the given number of trajectories from the given initial condition
    using the equations discovered by the given network (HyperSINDy or SINDy).

    Args:
        net: The network (nn.Module) to generate trajectories with. If
            HyperSINDy, generates trajectories with a randomly sampled set of
            equations at each timestep.
        device: The cpu or gpu device to generate trajetories with. If cpu,
            device must be the str "cpu". If gpu, device must be an int
            indicating which gpu to use (i.e. 0 or 1 or 2 or 3).
        x0: The initial condition used for each trajectory as a torch.Tensor
            of shape (z_dim,).
        batch_size: The number (int) of trajectories to generate.
        dt: The length (float) of one timestep to integrate forward in time.
        ts: The number of timesteps (int) to integrate.
    
    Returns:
        The generated trajectory as a numpy array of shape
        (batch_size x ts x z_dim).
    """
    zc = torch.from_numpy(x0).type(torch.FloatTensor).to(device)
    zc = torch.stack([zc for _ in range(batch_size)], dim=0)
    zs = []
    for i in range(ts):
        zc = zc + net(zc, device=device)[0] * dt
        zs.append(zc)
    zs = torch.stack(zs, dim=0)
    zs = torch.transpose(zs, 0, 1)
    return zs.detach().cpu().numpy()

def sample_ensemble_trajectory(net, device, x0, batch_size=10, dt=1e-2, ts=5000):
    """Generates trajectories.

    Generates the given number of trajectories from the given initial condition
    using the mean of the equations discovered by the given network (ESINDy).

    Args:
        net: The ESINDy network (nn.Module) to generate trajectories with.
        device: The cpu or gpu device to generate trajetories with. If cpu,
            device must be the str "cpu". If gpu, device must be an int
            indicating which gpu to use (i.e. 0 or 1 or 2 or 3).
        x0: The initial condition used for each trajectory as a torch.Tensor
            of shape (z_dim,).
        batch_size: The number (int) of trajectories to generate.
        dt: The length (float) of one timestep to integrate forward in time.
        ts: The number of timesteps (int) to integrate.
    
    Returns:
        The generated trajectory as a numpy array of shape
        (batch_size x ts x z_dim).
    """
    zc = torch.from_numpy(x0).type(torch.FloatTensor).to(device)
    zc = torch.stack([zc for _ in range(batch_size)], dim=0)
    zs = []
    coefs = net.get_masked_coefficients().mean(0)
    for i in range(ts):
        lib = net.make_library(zc)
        zc = zc + torch.matmul(lib, coefs) * dt
        zs.append(zc)
    zs = torch.stack(zs, dim=0)
    zs = torch.transpose(zs, 0, 1)
    return zs.detach().cpu().numpy()

def build_equation(lib, coef, eq):
    """Builds an equations.

    Builds an equation using the given coefficient and library terms and then
    appends the result to the given equation.

    Args:
        lib: A list of strings of each term in the SINDy library. Should 
            be what is returned from "equation_sindy_library" in
            model_utils.py.
        coef: The coefficients (numpy array of shape (library_dim,)) of
            each term in the library 
        eq: A string of the start of the equation to be created. For example,
            if eq = "dx = ", then appends the result to the right side of that
            string.

    Returns:
        A string of the created equation.
    """
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
    """Updates the list of equations.

    Appends equations corresponding to the given library and coefficients to
    given the list of equations.

    Args:
        equations: The list of strings to append equations to.
        library: A list of strings of each term in the SINDy library. Should 
            be what is returned from "equation_sindy_library" in
            model_utils.py.
        coefs: The coefficients (numpy array of shape library_dim x z_dim) of
            each term in the library
        starts: A list (of length z_dim) of strings denoting the start of each
            each equation. The string at index i in starts should correspond to
            z_dim[i], where 0 <= i < z_dim.
        z_dim: The number (int) of spatial dimensions in the data.

    Returns:
        None
    """
    for i in range(z_dim):
        equations.append(build_equation(library, coefs[:,i], starts[i]))

def get_equations(net, model_type, device, z_dim, poly_order, include_constant, include_sine):
    """Gets the equations learned by the network.

    Gets a list of the equations learned by the network. For HyperSINDy and
    ESINDy, gets both the mean and standard deviation over the coefficients.

    Args:
        net: The network (torch.nn.Module) to get the equations for.
        model_type: The str name of the model ({"HyperSINDy", "ESINDy",
            "SINDy"}). Equivalent to the model_type arguments in parse_args
            from cmd_line.py
        device: The cpu or gpu device to get the equations with. To use cpu,
            device must be "cpu". To use, specify which gpu as an integer
            (i.e.: 0 or 1 or 2 or 3).
        z_dim: The number (int) of spatial dimensions in the data.
        poly_order: The maximum order (int) of the polynomials in the library.
        include_constant: Iff True (boolean), includes a constant term in the library.
        include_sine: Iff True (boolean), includes sine in the library.

    Returns:
        The equations as a list of strings. For HyperSINDy and ESINDy,
        returns a list in the format:
            ["MEAN",
                equation_1,
                equation_2,
                ...,
                equation_n,
                "STD",
                equation_1,
                equation_2,
                ...,
                equation_n]
        where n = z_dim. For SINDy, returns a list in the format:
            ["SINDy",
                equation_1,
                equation_2,
                ...,
                equation_n]
        where n = z_dim.
    """
    starts = ["dx = ", "dy = ", "dz = "]
    if z_dim == 5:
        starts = ["dx1' = ", "dx2' = ", "dx3' = ", 'dx4 = ', 'dx5 = ']
    library = equation_sindy_library(z_dim, poly_order, include_constant=include_constant, include_sine=include_sine)
    equations = []
    if model_type == "HyperSINDy" or model_type == "ESINDy":
        mean_coeffs, std_coeffs = sindy_coeffs_stats(net.get_masked_coefficients(device=device))
        equations.append("MEAN")
        update_equation_list(equations, library, mean_coeffs, starts, z_dim)
        equations.append("STD")
        update_equation_list(equations, library, std_coeffs, starts, z_dim)
    elif model_type == "SINDy":
        sindy_coefs = net.get_masked_coefficients().detach().cpu().numpy()
        equations.append("SINDy")
        update_equation_list(equations, library, sindy_coefs, starts, z_dim)
    return equations


def sindy_coeffs_stats(sindy_coeffs):
    """Calculates the coefficient statistics.

    Calculates the mean and standard deviation of the given sindy coefficients
    along the batch dimension.

    Args:
        sindy_coeffs: The sindy coefficients as a torch.Tensor of shape
            (batch_size x library_dim x z_dim).

    Returns:
        A tuple of (array_a, array_b) where array_a is a numpy array of
        the mean of the coefficients and tensor_b is a numpy array of the
        standard deviation.
    """
    coefs = sindy_coeffs.detach().cpu().numpy()
    return np.mean(coefs, axis=0), np.std(coefs, axis=0)
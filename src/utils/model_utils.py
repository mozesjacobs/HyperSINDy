import torch
import torch.nn as nn
import numpy as np
from scipy.special import binom
from scipy.integrate import odeint


# Code taken from:
# https://github.com/kpchamp/SindyAutoencoders/blob/master/src/sindy_utils.py


def library_size(n, poly_order, use_sine=False, include_constant=True):
    l = 0
    for k in range(poly_order+1):
        l += int(binom(n+k-1,k))
    if use_sine:
        l += n
    if not include_constant:
        l -= 1
    return l


def sindy_library(X, poly_order=3, include_sine=False, include_constant=True):
    # batch x latent dim
    m, n = X.shape
    device = X.device
    l = library_size(n, poly_order, include_sine, include_constant)
    library = torch.ones((m,l), device=device)
    index = 0
    if include_constant:
        index = 1

    for i in range(n):
        library[:,index] = X[:,i]
        index += 1

    if poly_order > 1:
        for i in range(n):
            for j in range(i,n):
                library[:,index] = X[:,i] * X[:,j]
                index += 1

    if poly_order > 2:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    library[:,index] = X[:,i] * X[:,j] * X[:,k]
                    index += 1

    if poly_order > 3:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        library[:,index] = X[:,i] * X[:,j] * X[:,k] * X[:,q]
                        index += 1
                    
    if poly_order > 4:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        for r in range(q,n):
                            library[:,index] = X[:,i] * X[:,j] * X[:,k] * X[:,q] * X[:,r]
                            index += 1

    if include_sine:
        for i in range(n):
            library[:,index] = torch.sin(X[:,i])
            index += 1

    return library


def equation_sindy_library(n=3, poly_order=3, device=1, include_sine=False, include_constant=True):
    # timesteps x latent dim
    l = library_size(n, poly_order, include_sine, include_constant)
    index = 1
    X = ['x', 'y', 'z']
    str_lib = ['1']
    
    for i in range(n):
        str_lib.append(X[i])
    
    if poly_order > 1:
        for i in range(n):
            for j in range(i,n):
                str_lib.append(X[i] + X[j])
    
    if poly_order > 2:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    str_lib.append(X[i] + X[j] + X[k])

    if poly_order > 3:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        str_lib.append(X[i] + X[j] + X[k] + X[q])
    
    if poly_order > 4:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        for r in range(q,n):
                            str_lib.append(X[i] + X[j] + X[k] + X[q] + X[r])

    if include_sine:
        for i in range(n):
            str_lib.append('sin(' + X[i] + ')')

    return str_lib


def get_equation(lib, coef, start):
    res = start
    for i in range(len(coef)):
        if coef[i] != 0:
            res += str(coef[i]) + lib[i] + ' + '
    return res[:-2]


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def load_batch(data_set, batch, device):
    if data_set == 'arousal1':
        return load_arousal1_batch(batch, device)
    elif data_set == 'pendulum':
        return load_pendulum_batch(batch, device)
    elif data_set == 'delay_lorenz':
        return load_delay_lorenz_batch(batch, device)
    elif data_set == 'stochastic_lorenz':
        return load_stochastic_lorenz_batch(batch, device)
    elif data_set == 'stochastic_lorenz_flex':
        return load_stochastic_lorenz_flex_batch(batch, device)
    elif data_set == 'delay_lorenz_flex':
        return load_delay_lorenz_flex_batch(batch, device)
    elif data_set == 'hankel_lorenz':
        return load_hankel_lorenz_batch(batch, device)
    elif data_set == 'delay_lorenz_big':
        return load_delay_lorenz_big_batch(batch, device)
    elif data_set == 'delay_lorenz_simple':
        return load_delay_lorenz_simple_batch(batch, device)
    elif data_set == 'delay_lorenz_noise_simple':
        return load_delay_lorenz_noise_simple_batch(batch, device)
    elif data_set == 'delay_lorenz_simple2':
        return load_delay_lorenz_simple2_batch(batch, device)
    elif data_set == 'delay_lorenz_simple_all':
        return load_delay_lorenz_simple_all_batch(batch, device)
    elif data_set == 'lorenz2':
        return load_delay_lorenz2_batch(batch, device)
    elif data_set == 'lorenz_simple':
        return load_lorenz_simple_batch(batch, device)
    elif data_set == 'lorenz_simple_derivative':
        return load_lorenz_simple_derivative_batch(batch, device)


def load_arousal1_batch(batch, device):
    x1, x2 = batch
    x1 = x1.type(torch.FloatTensor).to(device)
    x2 = x2.type(torch.FloatTensor).to(device)
    x1_next = x1
    x2_next = x2
    return x1, x2, x1_next, x2_next


def load_delay_lorenz_batch(batch, device):
    (x1, x1_next, x2, x2_next), (z, z_next) = batch
    x1 = x1.reshape(x1.size(0), -1).type(torch.FloatTensor).to(device)
    x2 = x2.reshape(x2.size(0), -1).type(torch.FloatTensor).to(device)
    x1_next = x1_next.reshape(x1.size(0), -1).type(torch.FloatTensor).to(device)
    x2_next = x2_next.reshape(x2.size(0), -1).type(torch.FloatTensor).to(device)
    return (x1, x1_next, x2, x2_next), (z, z_next)

def load_stochastic_lorenz_batch(batch, device):
    (x1, x1_next, x2, x2_next), (z, z_next) = batch
    x1 = x1.reshape(x1.size(0), -1).type(torch.FloatTensor).to(device)
    x2 = x2.reshape(x2.size(0), -1).type(torch.FloatTensor).to(device)
    x1_next = x1_next.reshape(x1.size(0), -1).type(torch.FloatTensor).to(device)
    x2_next = x2_next.reshape(x2.size(0), -1).type(torch.FloatTensor).to(device)
    return (x1, x1_next, x2, x2_next), (z, z_next)

def load_stochastic_lorenz_flex_batch(batch, device):
    (x1, x1_next, x2, x2_next), (z, z_next), taus = batch
    x1 = x1.reshape(x1.size(0), -1).type(torch.FloatTensor).to(device)
    x2 = x2.reshape(x2.size(0), -1).type(torch.FloatTensor).to(device)
    x1_next = x1_next.reshape(x1.size(0), -1).type(torch.FloatTensor).to(device)
    x2_next = x2_next.reshape(x2.size(0), -1).type(torch.FloatTensor).to(device)
    taus = taus.type(torch.FloatTensor).to(device)
    return (x1, x1_next, x2, x2_next), (z, z_next), taus

def load_delay_lorenz_flex_batch(batch, device):
    (x1, x1_next, x1_next_next, x2, x2_next, x2_next_next), (z, z_next, z_next_next) = batch
    x1 = x1.reshape(x1.size(0), -1).type(torch.FloatTensor).to(device)
    x2 = x2.reshape(x2.size(0), -1).type(torch.FloatTensor).to(device)
    x1_next = x1_next.reshape(x1.size(0), -1).type(torch.FloatTensor).to(device)
    x2_next = x2_next.reshape(x2.size(0), -1).type(torch.FloatTensor).to(device)
    x1_next_next = x1_next_next.reshape(x1.size(0), -1).type(torch.FloatTensor).to(device)
    x2_next_next = x2_next_next.reshape(x2.size(0), -1).type(torch.FloatTensor).to(device)
    return (x1, x1_next, x1_next, x2, x2_next, x2_next), (z, z_next, z_next_next)

def load_hankel_lorenz_batch(batch, device):
    (x1, x1_next, x2, x2_next), (z, z_next) = batch
    x1 = x1.reshape(x1.size(0), -1).type(torch.FloatTensor).to(device)
    x2 = x2.reshape(x2.size(0), -1).type(torch.FloatTensor).to(device)
    x1_next = x1_next.reshape(x1.size(0), -1).type(torch.FloatTensor).to(device)
    x2_next = x2_next.reshape(x2.size(0), -1).type(torch.FloatTensor).to(device)
    return (x1, x1_next, x2, x2_next), (z, z_next)

def load_delay_lorenz_big_batch(batch, device):
    return load_delay_lorenz_batch(batch, device)

def load_lorenz2_batch(batch, device):
    (x1, x1_next, x2, x2_next), (z, z_next) = batch
    x1 = x1.reshape(x1.size(0), -1).type(torch.FloatTensor).to(device)
    x2 = x2.reshape(x2.size(0), -1).type(torch.FloatTensor).to(device)
    x1_next = x1_next.reshape(x1.size(0), -1).type(torch.FloatTensor).to(device)
    x2_next = x2_next.reshape(x2.size(0), -1).type(torch.FloatTensor).to(device)
    z = z.reshape(z.size(0), -1).type(torch.FloatTensor).to(device)
    z_next = z_next.reshape(z.size(0), -1).type(torch.FloatTensor).to(device)
    return (x1, x1_next, x2, x2_next), (z, z_next)

def load_delay_lorenz_simple_batch(batch, device):
    (_), (z, z_next) = batch
    z = z.type(torch.FloatTensor).to(device)[:, :, 0]
    z_next = z_next.type(torch.FloatTensor).to(device)[:, :, 0]
    #z = z.type(torch.FloatTensor).to(device)[:, :, 1]
    #z_next = z_next.type(torch.FloatTensor).to(device)[:, :, 1]
    return (None, None, None, None), (z, z_next)

def load_delay_lorenz_simple_all_batch(batch, device):
    (_), (z, z_next) = batch
    z = z.type(torch.FloatTensor).to(device).view(z.size(0), -1)
    z_next = z_next.type(torch.FloatTensor).to(device).view(z_next.size(0), -1)
    return (None, None, None, None), (z, z_next)

def load_delay_lorenz_simple2_batch(batch, device):
    (_), (z, z_next, z_next2) = batch
    z = z.type(torch.FloatTensor).to(device)[:, :, 0]
    z_next = z_next.type(torch.FloatTensor).to(device)[:, :, 0]
    z_next2 = z_next2.type(torch.FloatTensor).to(device)[:, :, :, 0]
    return (None, None, None, None), (z, z_next, z_next2)

def load_lorenz_simple_batch(batch, device):
    _, (z, z_next) = batch
    z = z.type(torch.FloatTensor).to(device)
    z_next = z_next.type(torch.FloatTensor).to(device)
    return (None, None, None, None), (z, z_next)

def load_delay_lorenz_noise_simple_batch(batch, device):
    (_), (z, z_next) = batch
    z = z.type(torch.FloatTensor).to(device)[:, :, 0]
    z_next = z_next.type(torch.FloatTensor).to(device)[:, :, 0]
    return (None, None, None, None), (z, z_next)

def load_lorenz_simple_derivative_batch(batch, device):
    _, (z, dz, z_next) = batch
    z = z.type(torch.FloatTensor).to(device)
    dz = dz.type(torch.FloatTensor).to(device)
    z_next = z_next.type(torch.FloatTensor).to(device)
    return (None, None, None, None), (z, dz, z_next)

def load_pendulum_batch(batch, device):
    (x, x_next), (z, ) = batch
    x = x.type(torch.FloatTensor).to(device)
    x_next = x_next.type(torch.FloatTensor).to(device)
    z = z.type(torch.FloatTensor).to(device)
    return (x, x_next), (z, )
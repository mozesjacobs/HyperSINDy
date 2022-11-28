import torch
import torch.nn as nn
import numpy as np
from scipy.special import binom
from scipy.integrate import odeint


# Code taken from:
# https://github.com/kpchamp/SindyAutoencoders/blob/master/src/sindy_utils.py

def library_size(n, poly_order, use_sine=False, use_mult_sine=False, include_constant=True):
    l = 0
    for k in range(poly_order+1):
        l += int(binom(n+k-1,k))
    if use_sine:
        l += n
    if use_mult_sine:
        l += n
    if not include_constant:
        l -= 1
    return l

def sindy_library(X, poly_order=3, include_sine=False, include_mult_sine=False, include_constant=True):
    # batch x latent dim
    m, n = X.shape
    device = X.device
    l = library_size(n, poly_order, include_sine, include_mult_sine, include_constant)
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

    if include_mult_sine:
        for i in range(n):
            library[:,index] = X[:,i] * torch.sin(X[:,i])
            index += 1

    return library
    
def equation_sindy_library(n=3, poly_order=3, device=1, include_sine=False, include_mult_sine=False, include_constant=True):
    # timesteps x latent dim
    l = library_size(n, poly_order, include_sine, include_constant)
    str_lib = []
    if include_constant:
        index = 1
        str_lib = ['']
    X = ['x', 'y', 'z']
    
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

    if include_mult_sine:
        for i in range(n):
            str_lib.append(X[i] + 'sin(' + X[i] + ')')

    return str_lib  

def init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform(layer.weight)
    elif isinstance(layer, nn.LayerNorm):
        layer.bias.data.fill_(0.01)
    elif isinstance(layer, nn.BatchNorm1d):
        layer.bias.data.fill_(0.01)
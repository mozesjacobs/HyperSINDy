import torch
import torch.nn as nn
import numpy as np
from scipy.special import binom
from scipy.integrate import odeint


def library_size(n, poly_order, use_sine=False, include_constant=True):
    """Calculates the size of the SINDy library.

    Calculates the number of terms in the SINDy library using the given
    parameters.

    The code for this function was taken from:
    https://github.com/kpchamp/SindyAutoencoders/blob/master/src/sindy_utils.py

    Args:
        n: The spatial dimenion (int) of the library.
        poly_order: The maximum degree of the polynomials to include in the
            the library. Includes integer polynomials from 1 up to and
            and including poly_order. Maximum value of poly_order is 5.
        use_sine: Iff True (boolean), includes sine in the library. The default
            is False.
        include_constant: Iff True (boolean), includes a constant term in the
            library. The default is True.

    Returns:
        The number of terms (int) in the library.
    """
    l = 0
    for k in range(poly_order+1):
        l += int(binom(n+k-1,k))
    if use_sine:
        l += n
    if not include_constant:
        l -= 1
    return l

def sindy_library(X, poly_order=3, include_sine=False, include_constant=True):
    """Creates a SINDy library.

    Creates a SINDy library out of X using the given settings.

    The code for this function was taken from:
    https://github.com/kpchamp/SindyAutoencoders/blob/master/src/sindy_utils.py

    Args:
        X: The data (torch.Tensor of shape (batch_size x z_dim)) to build a
            SINDy library with.
        poly_order: The maximum degree of the polynomials to include in the
            the library. Includes integer polynomials from 1 up to and
            and including poly_order. Maximum value of poly_order is 5.
        include_sine: Iff True (boolean), includes sine in the library. The default
            is False.
        include_constant: Iff True (boolean), includes a constant term in the
            library. The default is True.

    Returns:
        The SINDy library of X as a torch.Tensor of shape
        (batch_size x library_dim).
    """
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
    
def equation_sindy_library(n=3, poly_order=3, include_sine=False, include_constant=True):
    """Creates an equation SINDy library.

    Creates an equation SINDy library with the given settings. For n = 3, the
    result could be a list of the form:
        ["1", "x", "y", "z", "x^2", "xy", ...]
    The terms in the library should correspond to the terms returned by
    sindy_library, but represented as strings instead of the actual floats.

    Args:
        n: The spatial dimenion (int) of the library.
        poly_order: The maximum degree of the polynomials to include in the
            the library. Includes integer polynomials from 1 up to and
            and including poly_order. Maximum value of poly_order is 5.
        include_sine: Iff True (boolean), includes sine in the library. The default
            is False.
        include_constant: Iff True (boolean), includes a constant term in the
            library. The default is True.

    Returns:
        The SINDy library of X as a torch.Tensor of shape
        (batch_size x library_dim).
    """
    l = library_size(n, poly_order, include_sine, include_constant)
    str_lib = []
    if include_constant:
        index = 1
        str_lib = ['']
    
    X = ['x', 'y', 'z']
    if n == 5: 
        X = ['x1', 'x2', 'x3', 'x4', 'x5']
    
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

def init_weights(layer):
    """Initializes the weights.

    Initializes the weights of the layer. For Linear laters, uses xavier
    uniform initialization. For LayerNorm and BatchNorm1d layers, only 
    initializes the bias terms with the value 0.01. 

    Args:
        layer: The layer (nn.Linear, nn.LayerNorm, nn.BatchNorm1d) to
        initialize.
    
    Returns:
        None
    """
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform(layer.weight)
    elif isinstance(layer, nn.LayerNorm):
        layer.bias.data.fill_(0.01)
    elif isinstance(layer, nn.BatchNorm1d):
        layer.bias.data.fill_(0.01)
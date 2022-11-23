import torch
import torch.nn as nn
import numpy as np

class HyperNet(nn.Module):
    def __init__(self, in_dim, out_shape, hidden_dims=[8, 16, 32], bias=True, activation=nn.ELU(), norm='batch'):
        super(HyperNet, self).__init__()

        self.in_dim = in_dim
        self.out_shape = out_shape

        layers = []
        in_features = self.in_dim
        for out_features in hidden_dims:
            layers.append(nn.Linear(in_features, out_features, bias=bias))
            if norm == 'batch':
                layers.append(nn.BatchNorm1d(out_features))
            elif norm == 'layer':
                layers.append(nn.LayerNorm(out_features))
            layers.append(activation)
            in_features = out_features
        layers.append(nn.Linear(in_features, np.prod(self.out_shape), bias=bias))
        self.layers = nn.Sequential(*layers)

    def forward(self, n):
        return self.layers(n).reshape(n.size(0), *self.out_shape)
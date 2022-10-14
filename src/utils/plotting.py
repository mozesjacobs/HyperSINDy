import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def clear_ax_extra(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_aspect('equal')

def draw_hyperparams(ax, hyperparams):
    x_pos = (0.25, 0.25, 0.25, 0.25)
    y_pos = (0.8, 0.7, 0.6, 0.5)
    #ax.text(0.25, 0.90, "HYPERPARAMETERS", bbox=dict(facecolor='red', alpha=0.5))
    ax.text(0.25, 0.90, "HYPERPARAMETERS", fontweight='bold')
    names = ["l1 = ", "l2 = ", "l3 = ", "l4 = "]
    for idx, param in enumerate(hyperparams):
        ax.text(x_pos[idx], y_pos[idx], names[idx] + str(param))
    clear_ax_extra(ax)

def draw_equations(ax, equations, z_dim=3):
    x_pos = 0.25
    y_pos = 0.9
    dy = 0.055
    eq_interval = z_dim + 1
    for idx, eq in enumerate(equations):
        if idx % eq_interval == 0:
            #ax.text(x_pos, y_pos, eq, bbox=dict(facecolor='red', alpha=0.5))
            #ax.text(x_pos, y_pos, eq, color='red')
            ax.text(x_pos, y_pos, eq, color='red', fontweight='bold')
        else:
            #ax.text(x_pos, y_pos, eq, fontsize='small')
            ax.text(x_pos, y_pos, eq)
        y_pos -= dy
    clear_ax_extra(ax)

# zs shape: (num hyperparams x exp batch_size x trajectory length x 3)
def plot_3d_trajectory(fpath, zs, hyperparams, equations, figsize=None):
    num_hps, ebs, T, z_dim = zs.shape
    num_cols = ebs + 2 # first col = hyperparams, then exp batch, then equations
    if figsize is None:
        fig = plt.figure(figsize=(4 * ebs, (10 / 3) * num_hps), dpi=300)
    else:
        fig = plt.figure(figsize=figsize, dpi=300)
    ct = 1
    for i in range(num_hps):
        for j in range(-1, ebs + 1):
            # first column: hyperparams
            if j == -1:
                ax = fig.add_subplot(num_hps, num_cols, ct)
                draw_hyperparams(ax, hyperparams[i])
            # last column: equations
            elif j == ebs:
                ax = fig.add_subplot(num_hps, num_cols, ct)
                draw_equations(ax, equations[i], z_dim=z_dim)
            # in between: trajectories
            else:
                ax = fig.add_subplot(num_hps, ebs + 2, ct, projection='3d')
                ax.plot(zs[i][j][:,0], zs[i][j][:,1], zs[i][j][:,2])
            ct += 1
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.savefig(fpath)
    plt.close()

# zs shape: (num hyperparams x exp batch_size x trajectory length x 1)
def plot_1d_trajectory(fpath, z_true, zs, hyperparams, equations, figsize=None):
    num_hps, ebs, T, z_dim = zs.shape
    num_cols = ebs + 2 # first col = hyperparams, then exp batch, then equations
    if figsize is None:
        fig = plt.figure(figsize=(4 * ebs, (10 / 3) * num_hps), dpi=300)
    else:
        fig = plt.figure(figsize=figsize, dpi=300)
    ct = 1
    for i in range(num_hps):
        for j in range(-1, ebs + 1):
            # first column: hyperparams
            if j == -1:
                ax = fig.add_subplot(num_hps, num_cols, ct)
                draw_hyperparams(ax, hyperparams[i])
            # last column: equations
            elif j == ebs:
                ax = fig.add_subplot(num_hps, num_cols, ct)
                draw_equations(ax, equations[i], z_dim=z_dim)
            # in between: trajectories
            else:
                ax = fig.add_subplot(num_hps, ebs + 2, ct)
                if (i == num_hps - 1) and (j == ebs - 1):
                    ax.plot(zs[i][j][:,0], color='red', label='P')
                    ax.plot(z_true, color='blue', label='GT')
                    ax.legend(loc='best')
                else:
                    ax.plot(zs[i][j][:,0], color='red')
                    ax.plot(z_true, color='blue')
            ct += 1
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.savefig(fpath)
    plt.close()
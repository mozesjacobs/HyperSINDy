import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def clear_ax_extra(ax):
    """Clears the axis.

    Clears the ticks and spines of this given axis and sets its aspect to
    equal.

    Args:
        ax: The axis (plt.axes) to modify.
    
    Returns:
        None
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_aspect('equal')


def draw_equations(board, epoch, equations, z_dim):
    """Draws the equations.

    Draws the given equations on a figure, then adds that figure to the given
    board.

    Args:
        board: The SummaryWriter to log the figure with.
        epoch: The epoch (int) to log the figure at.
        equations: A list of equations (strings) to draw on the board. Should
            be what is returned by "get_equations" in exp_utils.py.
        z_dim: The spatial dimension (int) of the data.
    """
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.gca()
    x_pos = 0.25
    y_pos = 0.9
    dy = 0.055
    for idx, eq in enumerate(equations):
        if idx % (z_dim + 1) == 0:
            ax.text(x_pos, y_pos, eq, color='red', fontweight='bold')
        else:
            ax.text(x_pos, y_pos, eq)
        y_pos -= dy
    clear_ax_extra(ax)
    board.add_figure(tag="Equations", figure=fig, global_step=epoch, close=True,
                     walltime=None)


def plot_trajectory(board, epoch, z_true, z_pred, figsize=None):
    """Plots the trajectories.

    Plots the given ground truth and predicted trajectories, then adds the
    the figure to the tensorboard.

    If z_dim = 1, plots the trajectory on the y axis, with the x axis being
    time.
    
    If z_dim = 2, plots the spatial dimensions of the trajectory
    separately on the y axis, with x still being time.

    If z_dim = 3, creates a 3D plot of the trajectory.

    If z_dim = 5, creates a 3D plot using the first 3 spatial dimensions of the
    trajectory.

    Args:
        board: The SummaryWriter to log the figure with.
        epoch: The epoch (int) to log the figure at.
        z_true: The ground truth trajectory as numpy array of shape
            (timesteps x z_dim).
        z_pred: The generated trajectories as a numpy array of shape
            (batch_size x timesteps x z_dim)
        figsize: The size of the figure (float, float) to plot. Default is
            None. If None, sets figsize to (batch_size x 3.5).

    Returns:
        None
    """
    batch_size, T, z_dim = z_pred.shape
    if figsize is None:
        fig = plt.figure(figsize=(batch_size, 3.5), dpi=300)
    else:
        fig = plt.figure(figsize=figsize, dpi=300)
    for i in range(batch_size):
        if z_dim == 1:
            ax = fig.add_subplot(1, batch_size, i + 1)
            ax.plot(z_pred[i, :,0][j], color='red', label='Pred')
            ax.plot(z_true, color='blue', label='GT')
            ax.legend(loc='best')
        elif z_dim == 2:
            ax = fig.add_subplot(1, batch_size, i + 1)
            ax.plot(z_pred[i, :,0], color='red', label='X Pred')
            ax.plot(z_pred[i, :,1], color='blue', label='Y Pred')
            ax.plot(z_true[:,0], color='yellow', label='X GT')
            ax.plot(z_true[:,1], color='green', label='Y GT')
            ax.legend(loc='best')
        elif z_dim == 3:
            if i == 0:
                ax = fig.add_subplot(1, batch_size + 1, i + 1, projection='3d')
                ax.plot(z_true[:,0], z_true[:,1], z_true[:,2], color='red', label="GT")
            ax = fig.add_subplot(1, batch_size + 1, i + 2, projection='3d')
            ax.plot(z_pred[i, :,0], z_pred[i, :,1], z_pred[i, :,2], color='blue', label="Pred")
        elif z_dim == 5:
            if i == 0:
                ax = fig.add_subplot(1, batch_size + 1, i + 1, projection='3d')
                ax.plot(z_true[:,0], z_true[:,1], z_true[:,2], color='red', label="GT")
            ax = fig.add_subplot(1, batch_size + 1, i + 2, projection='3d')
            ax.plot(z_pred[i, :,0], z_pred[i, :,1], z_pred[i, :,2], color='blue', label="Pred")
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    board.add_figure(tag="Samples", figure=fig, global_step=epoch, close=True,
                     walltime=None)
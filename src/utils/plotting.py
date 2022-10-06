import imageio
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
#import seaborn as sns
import math


def canvas2rgb_array(canvas):
    # https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
    """Adapted from: https://stackoverflow.com/a/21940031/959926"""
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    ncols, nrows = canvas.get_width_height()
    scale = int(round(math.sqrt(buf.size / 3 / nrows / ncols)))
    return buf.reshape(scale * nrows, scale * ncols, 3)


def get_fig(x1, x1_pred, x2, x2_pred, num, ylims1, ylims2):
    fig, axes = plt.subplots(num, 2, figsize=(5, 5))
    for i in range(num):
        axes[i][0].plot(x1[i], color='blue')
        axes[i][0].plot(x1_pred[i], color='red')
        axes[i][1].plot(x2[i], color='blue')
        axes[i][1].plot(x2_pred[i], color='red')
        axes[i][0].set_ylim([ylims1[0][i], ylims1[1][i]])
        axes[i][1].set_ylim([ylims2[0][i], ylims2[1][i]])
    plt.subplots_adjust(wspace=0)
    plt.close()
    return fig


def make_gif(x1, x1_pred, x2, x2_pred, T, num, fpath, duration=0.1):
    x1 = np.swapaxes(x1, 0, 1)
    x1_pred = np.swapaxes(x1_pred, 0, 1)
    x2 = np.swapaxes(x2, 0, 1)
    x2_pred = np.swapaxes(x2_pred, 0, 1)
    images = []
    y_min1, y_max1 = np.min(x1, axis=0), np.max(x1, axis=0)
    y_min1, y_max1 = np.min(y_min1, axis=1), np.max(y_max1, axis=1)
    y_min2, y_max2 = np.min(x2, axis=0), np.max(x2, axis=0)
    y_min2, y_max2 = np.min(y_min2, axis=1), np.max(y_max2, axis=1)
    for i in range(T):
        the_fig = get_fig(x1[i], x1_pred[i], x2[i], x2_pred[i], num, (y_min1, y_max1), (y_min2, y_max2))
        data = canvas2rgb_array(the_fig.canvas)
        images.append(data)
    imageio.mimsave(fpath, images, duration=duration)

def plot_latent_trajectory(z, z_true, num_plot, fig_path, figsize=(10, 30)):
    fig = plt.figure(figsize=figsize)
    fig.tight_layout()
    start = 0
    for i in range(num_plot):
        ax = fig.add_subplot(num_plot, 2, i + 1, projection='3d')
        ax.view_init(60, 35)
        ax.plot(z[i][:,0][start:], z[i][:,1][start:], z[i][:,2][start:], color='red', label='Model')
        ax.plot(z_true[i][:,0][start:], z_true[i][:,1][start:], z_true[i][:,2][start:], color='blue', label='Ground Truth')
        ax.set_title("Trajectory " + str(i + 1))
    plt.legend(loc='best')
    if fig_path is not None:
        plt.savefig(fig_path)
        plt.close()
    else:
        plt.show()

def plot_hankel_latent_trajectory(z, z_true, fig_path, figsize=(10, 30), view_inits=None):
    # z_true : samples, comps, zdim
    fig = plt.figure(figsize=figsize)
    fig.tight_layout()
    ax = fig.add_subplot(1, 4, 1, projection='3d')
    if view_inits is not None:
        ax.view_init(view_inits[0], view_inits[1])
    ax.plot(z[:,0], z[:,1], z[:,2], color='red', label='Model')
    ax.set_title("Model")
    hankels = ['x', 'y', 'z']
    for i in range(z_true.shape[-1]):
        ax = fig.add_subplot(1, 4, i + 2, projection='3d')
        ax.plot(z_true[:, 0][:, i], z_true[:, 1][:, i], z_true[:, 2][:, i], color='blue')
        ax.set_title(hankels[i] + " embedding of hankel")
    if fig_path is not None:
        plt.savefig(fig_path)
        plt.close()
    else:
        plt.show()

def plot_single_latents(z, z_true, fig_path, figsize=(10, 30), view_inits=None):
    fig = plt.figure(figsize=figsize)
    fig.tight_layout()
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    if view_inits is not None:
        ax.view_init(view_inits[0], view_inits[1])
    start = 0
    ax.plot(z[:,0][start:], z[:,1][start:], z[:,2][start:], color='red', label='Model')
    ax.plot(z_true[:,0][start:], z_true[:,1][start:], z_true[:,2][start:], color='blue', label='Ground Truth')
    ax.set_title("Both")

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.plot(z[:,0][start:], z[:,1][start:], z[:,2][start:], color='red', label='Model')
    ax.set_title("Model")

    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.plot(z_true[:,0][start:], z_true[:,1][start:], z_true[:,2][start:], color='blue', label='Ground Truth')
    ax.set_title("Ground Truth")

    plt.legend(loc='best')
    if fig_path is not None:
        plt.savefig(fig_path)
        plt.close()
    else:
        plt.show()

def get_pendulum_fig(x1, x1_true):
    fig, axes = plt.subplots(1, 2, figsize=(5, 5))
    axes[0].imshow(x1, cmap='gray')
    axes[1].imshow(x1_true, cmap='gray')
    axes[0].set_title("pred")
    axes[1].set_title("ground truth")
    """
    for i in range(num):
        axes[i][0].plot(x1[i], color='blue')
        axes[i][0].plot(x1_pred[i], color='red')
        axes[i][1].plot(x2[i], color='blue')
        axes[i][1].plot(x2_pred[i], color='red')
        axes[i][0].set_ylim([ylims1[0][i], ylims1[1][i]])
        axes[i][1].set_ylim([ylims2[0][i], ylims2[1][i]])
    """
    plt.subplots_adjust(wspace=0)
    plt.close()
    return fig

def make_pendulum_gif(x, x_true, fpath, duration=0.1):
    #x1 = np.swapaxes(x1, 0, 1)
    #x1_pred = np.swapaxes(x1_pred, 0, 1)
    #x2 = np.swapaxes(x2, 0, 1)
    #x2_pred = np.swapaxes(x2_pred, 0, 1)
    #images = []
    #y_min1, y_max1 = np.min(x1, axis=0), np.max(x1, axis=0)
    #y_min1, y_max1 = np.min(y_min1, axis=1), np.max(y_max1, axis=1)
    #y_min2, y_max2 = np.min(x2, axis=0), np.max(x2, axis=0)
    #y_min2, y_max2 = np.min(y_min2, axis=1), np.max(y_max2, axis=1)
    T = min(x.shape[0], x_true.shape[0])
    images = []
    for i in range(T):
        the_fig = get_pendulum_fig(x[i], x_true[i])
        data = canvas2rgb_array(the_fig.canvas)
        images.append(data)
    imageio.mimsave(fpath, images, duration=duration)
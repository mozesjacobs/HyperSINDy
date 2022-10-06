import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from src.utils.other import *
from src.utils.plotting import make_gif, plot_latent_trajectory, plot_hankel_latent_trajectory, plot_single_latents, make_pendulum_gif

def exp1(args, net, save_path):
    # val data (will refer to the val data as test data)
    _, test_set, _ = load_data(args)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
    net.eval()
    batch = next(iter(test_loader))
    (x1, x1_next, x2, x2_next), (z_true, z_true_next) = batch
    _, (x1_hat, x2_hat, z) = net(batch, just_mean=exp_just_mean)
    #_, (x1_hat, x2_hat, x1_bar, x2_bar, z1, z2) = net(batch, just_mean=args.just_mean)
    x1_hat = x1_hat.reshape(-1, x1_next.size(1), args.u_dim).detach().cpu().numpy()
    x2_hat = x2_hat.reshape(-1, x2_next.size(1), args.u_dim).detach().cpu().numpy()
    #x1_bar = x1_bar.reshape(-1, x1_next.size(1), args.u_dim).detach().cpu().numpy()
    #x2_bar = x2_bar.reshape(-1, x2_next.size(1), args.u_dim).detach().cpu().numpy()
    make_gif(x1_next.numpy(), x1_hat, x2_next.numpy(), x2_hat, x1_hat.shape[1],
             args.num_plot, save_path, duration=args.gif_duration) 
    #make_gif(x1_next.numpy(), x1_bar, x2_next.numpy(), x2_bar, x1_hat.shape[1],
    #         args.num_plot, save_path, duration=args.gif_duration) 

def exp2(args, net, save_path):
    # val data (will refer to the val data as test data)
    _, test_set, _ = load_data(args)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
    net.eval()
    batch = next(iter(test_loader), just_mean=args.exp_just_mean)
    (x1, x1_next, x2, x2_next), (z_true, z_true_next) = batch
    #_, (x1_hat, x2_hat, z) = net(batch, just_mean=True)
    _, (x1_hat, x2_hat, x1_bar, x2_bar, z, z2) = net(batch, just_mean=args.just_mean)
    z = z.reshape(-1, z_true_next.size(1), args.z_dim).detach().cpu().numpy()
    z_true_next = z_true_next.reshape(-1, z_true_next.size(1), args.z_dim).detach().cpu().numpy()
    plot_latent_trajectory(z, z_true_next, args.num_plot, save_path)

def exp3(args, net, save_path):
    net.eval()
    # val data (will refer to the val data as test data)
    _, test_set, _ = load_data(args)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
    batch = next(iter(test_loader))
    (x1, x1_next, x2, x2_next), (z_true, z_true_next) = batch

    device = args.device
    x1 = x1[:, 0].type(torch.FloatTensor).to(device)
    x2 = x2[:, 0].type(torch.FloatTensor).to(device)

    z = net.sample_trajectory(args.timesteps, just_mean=args.exp_just_mean, x1=x1, x2=x2)
    z_true = z_true.detach().cpu().numpy()
    plot_latent_trajectory(z, z_true, args.num_plot, save_path)


def exp4(args, net, save_path):
    net.eval()
    device = args.device
    x1_hats, x2_hats = [], []
    z = torch.randn([args.batch_size, args.z_dim], device=device)
    z = net.sample_trajectory(args.exp_timesteps - 1, just_mean=args.exp_just_mean, z=z)
    plot_latent_trajectory(z, z, args.num_plot, save_path)


def exp5(args, net, save_path):
    net.eval()
    _, test_set, _ = load_data(args)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False, num_workers=1)
    batch = next(iter(test_loader))
    (x1, x1_next, x2, x2_next), (z_true, z_true_next) = batch
    _, (x1_hat, x2_hat, z) = net(batch, just_mean=args.exp_just_mean)
    z = z.detach().cpu().numpy()
    z_true_next = z_true_next.detach().cpu().numpy()
    plot_hankel_latent_trajectory(z, z_true_next, save_path, figsize=(10, 10))

def exp6(args, net, save_path):
    net.eval()
    device = args.device
    _, test_set, _ = load_data(args)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False, num_workers=1)
    batch = next(iter(test_loader))
    (x1, x1_next, x2, x2_next), (z_true, z_true_next) = batch
    x1 = x1[0:args.num_plot].reshape(args.num_plot, -1).type(torch.FloatTensor).to(device)[0].unsqueeze(0)
    x2 = x2[0:args.num_plot].reshape(args.num_plot, -1).type(torch.FloatTensor).to(device)[0].unsqueeze(0)
    #z = net.sample_trajectory(z_true_next.size(0) - 1, just_mean=args.exp_just_mean, x1=x1, x2=x2)[0]
    z = net.sample_trajectory(args.exp_timesteps, just_mean=args.exp_just_mean, x1=x1, x2=x2)[0]
    z_true_next = z_true_next.detach().cpu().numpy()
    start, end = args.z_plot_start, args.z_plot_end
    z = z[start:end]
    #print(z)
    z_true_next = z_true_next[start:end]
    view_inits = None
    if args.v1 is not None and args.v2 is not None:
        view_inits = (args.v1, args.v2)
    plot_hankel_latent_trajectory(z, z_true_next, save_path, figsize=(10, 10), view_inits=view_inits)

def exp7(args, net, save_path):
    net.eval()
    # val data (will refer to the val data as test data)
    _, test_set, _ = load_data(args)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False, num_workers=1)
    batch = next(iter(test_loader))
    _, (z_true, z_true_next) = batch
    #_, (z_true, z_true_next, z_true_next2) = batch

    device = args.device
    #z_init = z_true.type(torch.FloatTensor).to(device)[0].unsqueeze(0) # 1 x 3
    z_init = z_true.type(torch.FloatTensor).to(device)[:, :, 0][0].unsqueeze(0) # 1 x 3
    #z_init = z_true.type(torch.FloatTensor).to(device)[0].unsqueeze(0) # 1 x 3
    start, end = args.z_plot_start, args.z_plot_end
    if start is None:
        start = 0
    if end is None:
        end = 1 + args.exp_timesteps
    z = net.sample_trajectory(args.exp_timesteps, just_mean=args.exp_just_mean, z=z_init)[0][start:end]
    plot_hankel_latent_trajectory(z, z_true_next.detach().cpu().numpy(), save_path, figsize=(10, 10))

def exp8(args, net, save_path):
    net.eval()
    #print(net.sindy_coefficients)
    # val data (will refer to the val data as test data)
    _, test_set, _ = load_data(args)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False, num_workers=1)
    batch = next(iter(test_loader))
    _, (z_true, z_true_next) = batch

    device = args.device
    z_true = z_true.type(torch.FloatTensor).to(device)[0].unsqueeze(0) # 1x3

    z = net.sample_trajectory(args.timesteps, just_mean=args.exp_just_mean, z=z_true)[0]
    z_true_next = z_true_next.detach().cpu().numpy()
    start, end = args.z_plot_start, args.z_plot_end
    if start is None:
        start = 0
    if end is None:
        end = 1 + args.exp_timesteps
    print(z)
    #print(net.sindy_coefficients * net.threshold_mask)
    plot_single_latents(z[start:end], z_true_next, save_path)

def exp9(args, net, save_path):
    net.eval()
    device = args.device
    _, test_set, _ = load_data(args)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False, num_workers=1)
    batch = next(iter(test_loader))
    (x1, x1_next, x2, x2_next), (z_true, z_true_next) = batch
    #x1 = x1[0:args.num_plot].reshape(args.num_plot, -1).type(torch.FloatTensor).to(device)[0].unsqueeze(0)
    #x2 = x2[0:args.num_plot].reshape(args.num_plot, -1).type(torch.FloatTensor).to(device)[0].unsqueeze(0)
    _, (_, _, z) = net(batch, just_mean=args.exp_just_mean, return_post=args.exp_return_post)
    z = z.detach().cpu().numpy()
    z_true_next = z_true_next.detach().cpu().numpy()
    start, end = args.z_plot_start, args.z_plot_end
    z = z[start:end]
    z_true_next = z_true_next[start:end]
    view_inits = None
    if args.v1 is not None and args.v2 is not None:
        view_inits = (args.v1, args.v2)
    plot_hankel_latent_trajectory(z, z_true_next, save_path, figsize=(10, 10), view_inits=view_inits)

def exp10(args, net, save_path):
    net.eval()
    device = args.device
    _, test_set, _ = load_data(args)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False, num_workers=1)
    batch = next(iter(test_loader))
    (x1, x1_next, x2, x2_next), (z_true, z_true_next) = batch
    x1 = x1[0:args.num_plot].reshape(args.num_plot, -1).type(torch.FloatTensor).to(device)[0].unsqueeze(0)
    x2 = x2[0:args.num_plot].reshape(args.num_plot, -1).type(torch.FloatTensor).to(device)[0].unsqueeze(0)
    #z = net.sample_trajectory(z_true_next.size(0) - 1, just_mean=args.exp_just_mean, x1=x1, x2=x2)[0]
    z = net.sample_trajectory(args.exp_timesteps, just_mean=args.exp_just_mean, x1=x1, x2=x2)[0]
    z_true_next = z_true_next.detach().cpu().numpy()
    start, end = args.z_plot_start, args.z_plot_end
    z = z[start:end]
    #print(z)
    z_true_next = z_true_next[start:end]
    view_inits = None
    if args.v1 is not None and args.v2 is not None:
        view_inits = (args.v1, args.v2)
    else:
        view_inits = None
    plot_single_latents(z[start:end], z_true_next, save_path, view_inits=view_inits)

def exp11(args, net, save_path):
    net.eval()
    device = args.device
    _, test_set, _ = load_data(args)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False, num_workers=1)
    batch = next(iter(test_loader))
    (x1, x1_next, x1_nn, x2, x2_next, x2_nn), (z_true, z_true_next, z_nn) = batch
    x1 = x1[0:args.num_plot].reshape(args.num_plot, -1).type(torch.FloatTensor).to(device)[0].unsqueeze(0)
    x2 = x2[0:args.num_plot].reshape(args.num_plot, -1).type(torch.FloatTensor).to(device)[0].unsqueeze(0)
    #z = net.sample_trajectory(z_true_next.size(0) - 1, just_mean=args.exp_just_mean, x1=x1, x2=x2)[0]
    z = net.sample_trajectory(args.exp_timesteps, just_mean=args.exp_just_mean, x1=x1, x2=x2)[0]
    z_true_next = z_true_next.detach().cpu().numpy()
    start, end = args.z_plot_start, args.z_plot_end
    z = z[start:end]
    #print(z)
    z_true_next = z_true_next[start:end]
    view_inits = None
    if args.v1 is not None and args.v2 is not None:
        view_inits = (args.v1, args.v2)
    plot_hankel_latent_trajectory(z, z_true_next, save_path, figsize=(10, 10), view_inits=view_inits)

def exp12(args, net, save_path):
    net.eval()
    device = args.device
    _, test_set, _ = load_data(args)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False, num_workers=1)
    batch = next(iter(test_loader))
    (x1, x1_next), (z_true, ) = batch
    z_true = test_set.z.reshape(test_set.num_samples, test_set.timesteps)[0]
    x1 = x1[0:args.num_plot].type(torch.FloatTensor).to(device)[0].unsqueeze(0)
    #z = net.sample_trajectory(z_true_next.size(0) - 1, just_mean=args.exp_just_mean, x1=x1, x2=x2)[0]
    z = net.sample_trajectory(args.exp_timesteps, just_mean=args.exp_just_mean, x1=x1)[0]
    z_true = z_true.detach().cpu().numpy()
    start, end = args.z_plot_start, args.z_plot_end
    z = z[start:end]
    #print(z)
    z_true = z_true[start:end]
    view_inits = None
    if args.v1 is not None and args.v2 is not None:
        view_inits = (args.v1, args.v2)
    else:
        view_inits = None
    #print(z.shape)
    #print(z_true.shape)
    #"""
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(z_true, color='red')
    plt.plot(z, color='blue')
    plt.savefig(save_path)
    plt.close()
    #"""
    #plot_single_latents(z[start:end], z_true_next, save_path, view_inits=view_inits)

def exp13(args, net, save_path):
    # val data (will refer to the val data as test data)
    net.eval()
    device = args.device
    _, test_set, _ = load_data(args)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False, num_workers=1)
    batch = next(iter(test_loader))
    (x1, x1_next), (z_true, ) = batch
    #z_true = test_set.z.reshape(test_set.num_samples, test_set.timesteps)[0]
    x1 = x1[0:args.num_plot].type(torch.FloatTensor).to(device)[0].unsqueeze(0)
    z = net.sample_trajectory(args.exp_timesteps, just_mean=args.exp_just_mean, x1=x1)[0]
    x = net.decode(torch.from_numpy(z).type(torch.FloatTensor).to(device))
    x = x.detach().cpu().numpy()[1:-1].reshape(-1, test_set.x_dim, test_set.x_dim)
    x1 = test_set.x_next.detach().cpu().numpy().reshape(-1, test_set.timesteps - args.tau, test_set.x_dim, test_set.x_dim)
    x1 = x1[0]#.reshape(-1, test_set.x_dim, test_set.x_dim)
    #print(x.shape)
    #print(x1.shape)
    make_pendulum_gif(x, x1, save_path, duration=args.gif_duration)

def exp14(args, net, save_path):
    net.eval()
    device = args.device
    _, test_set, _ = load_data(args)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False, num_workers=1)
    batch = next(iter(test_loader))
    (x1, x1_next, x2, x2_next), (z_true, z_true_next), taus = batch
    x1 = x1[0:args.num_plot].reshape(args.num_plot, -1).type(torch.FloatTensor).to(device)[0].unsqueeze(0)
    x2 = x2[0:args.num_plot].reshape(args.num_plot, -1).type(torch.FloatTensor).to(device)[0].unsqueeze(0)
    taus = taus.type(torch.FloatTensor).to(device)[0].unsqueeze(0)
    z = net.sample_trajectory(args.exp_timesteps, taus * args.delta_t, just_mean=args.exp_just_mean, x1=x1, x2=x2)[0]
    #z = net.sample_trajectory(args.exp_timesteps, 10 * args.delta_t, just_mean=args.exp_just_mean, x1=x1, x2=x2)[0]
    z_true_next = z_true_next.detach().cpu().numpy()
    start, end = args.z_plot_start, args.z_plot_end
    z = z[start:end]
    #print(z)
    z_true_next = z_true_next[start:end]
    view_inits = None
    if args.v1 is not None and args.v2 is not None:
        view_inits = (args.v1, args.v2)
    plot_hankel_latent_trajectory(z, z_true_next, save_path, figsize=(10, 10), view_inits=view_inits)
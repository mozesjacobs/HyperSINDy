import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from src.utils.other import save_model
from src.utils.exp_utils import sample_trajectory, get_equations
from src.utils.plotting import draw_equations, plot_trajectory

def train(net, args, hyperparams, optim, scheduler, trainloader, trainset, 
          board, cp_path, initial_epoch, device): 
    
    beta = args.beta_init
    beta_max = hyperparams.beta
    beta_inc = args.beta_inc
    if beta_inc is None:
        beta_inc = beta_max / 100.0

    for epoch in range(initial_epoch, hyperparams.epochs + initial_epoch):
        # one train step
        recons, klds = train_epoch(
            net, args.model, trainloader, optim, beta,
            hyperparams.weight_decay, device, hyperparams.clip)

        # log losses
        log_losses(board, recons / len(trainloader), klds / len(trainloader), epoch)

        # threshold
        update_threshold_mask(net, args.model, hyperparams.threshold,
                              hyperparams.threshold_interval, epoch, device,
                              beta, beta_max)

        # save
        if (epoch % args.checkpoint_interval == 0) and (epoch != 0):
            save_model(cp_path, net, optim, scheduler, epoch)

        # save
        if (epoch % args.eval_interval == 0) and (epoch != 0):
            eval_model(net.eval(), args, board, trainset, device, epoch)

        scheduler.step()
        beta = update_beta(beta, beta_inc, beta_max)

    save_model(cp_path, net, optim, scheduler, epoch)
    eval_model(net.eval(), args, board, trainset, device, epoch)

    return net, optim, scheduler

def train_epoch(net, model_type, trainloader, optim, beta, weight_decay,
                device, clip):
    # train mode
    net = net.train()
    
    recons, klds = 0, 0
    for i, (x, x_lib, x_dot) in enumerate(trainloader):
        x_dot = x_dot.type(torch.FloatTensor).to(device)

        # one gradient step
        if model_type == "HyperSINDy":
            recon, kld = train_hyper(net, optim, x, x_lib, x_dot, beta,
                                     weight_decay, device, clip)
        elif model_type == "SINDy":
            recon, kld = train_sindy(net, optim, x, x_lib, x_dot, weight_decay,
                                     device, clip)

        recons += recon
        klds += kld
    return recons, klds 

def train_hyper(net, optim, x, x_lib, x_dot, beta, weight_decay, device, clip):
    x_dot_pred, sindy_coeffs = net(x, x_lib, device)
    recon = ((x_dot_pred - x_dot) ** 2).sum(1).mean()
    kld = net.kl(sindy_coeffs)
    loss = recon + kld * beta
    optim.zero_grad()
    loss.backward()
    if clip is not None:
        nn.utils.clip_grad_norm_(net.parameters(), clip)
    optim.step()
    return recon.item(), kld.item()


def train_sindy(net, optim, x, x_lib, x_dot, weight_decay, device, clip):
    x_dot_pred, sindy_coeffs = net(x, x_lib, device)
    recon = ((x_dot_pred - x_dot) ** 2).sum(1).mean()
    masked_coeffs = sindy_coeffs * net.threshold_mask
    regularization = (masked_coeffs ** 2).sum() * weight_decay
    loss = recon + regularization
    optim.zero_grad()
    loss.backward()
    if clip is not None:
        n.utils.clip_grad_norm_(net.parameters(), clip)
    optim.step()
    return recon, 0

def update_threshold_mask(net, model_type, threshold, threshold_timer, epoch, device, beta, beta_max):
    with torch.no_grad():
        if (epoch % threshold_timer == 0) and (epoch != 0) and (beta == beta_max):
            if (model_type == "HyperSINDy"):
                if (beta == beta_max):
                    net.update_threshold_mask(threshold, device)
            else:
                net.update_threshold_mask(threshold, device)

def log_losses(board, recon, kl, epoch):
    # tensorboard
    board.add_scalar("Loss/(x_dot_pred - x_dot) ** 2", recon, epoch)
    board.add_scalar("Loss/kld", kl, epoch)


def update_beta(beta, beta_increment, beta_max):
    beta += beta_increment
    if beta > beta_max:
        beta = beta_max
    return beta

def eval_model(net, args, board, trainset, device, epoch):
    # sample trajectory
    z = sample_trajectory(net, device, trainset.x[0].numpy(),
                          args.exp_batch_size, args.dt, args.exp_timesteps)

    # plot trajectory
    plot_trajectory(board, epoch, trainset.x.numpy(), z)

    # get equations
    equations = get_equations(net, args.model, device,
                              args.z_dim, args.poly_order,
                              args.include_constant, args.include_sine)

    eq_mean = str(equations[1]) + "  \n" + str(equations[2]) + "  \n" + str(equations[3])
    eq_std = str(equations[5]) + "  \n" + str(equations[6]) + "  \n" + str(equations[7])
    board.add_text(tag="Equations/mean", text_string=eq_mean, global_step=epoch, walltime=None)
    board.add_text(tag="Equations/std", text_string=eq_std, global_step=epoch, walltime=None)

    # Prior scale
    board.add_scalar("Prior Scale", net.prior_scale, epoch)

    #draw_equations(board, epoch, equations, args.z_dim)
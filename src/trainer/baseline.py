import torch
import numpy as np
from tqdm import tqdm
from src.utils.other import save_model

def train(net, optim, scheduler, cp_path, model_type, trainloader, board,
          device, initial_epoch, epochs, beta_init, beta_increment,
          checkpoint_interval, threshold_interval, beta_max,
          weight_decay, threshold): 

    if beta_increment is None:
        beta_increment = beta_max / 100.0
    beta = 0
    for e in range(initial_epoch, epochs + initial_epoch):
        # one train step
        recons, klds = train_epoch(net, model_type, trainloader, optim, beta, weight_decay, device)

        # log losses
        log_losses(board, recons / len(trainloader), klds / len(trainloader), e)

        # threshold
        update_threshold_mask(net, threshold, threshold_interval, e, device, beta, beta_max):

        # save
        if (e + 1) % checkpoint_interval == 0:
            save_model(cp_path, net, optim, scheduler, e)

        scheduler.step()
        beta = update_beta(beta, beta_increment, beta_max)

    save_model(cp_path, net, optim, scheduler, e)

    return net, optim, scheduler

def train_epoch(net, model_type, trainloader, optim, beta, weight_decay, device):
    # train mode
    net = net.train()
    
    recons, klds = 0, 0
    for i, (x, x_dot) in enumerate(trainloader):
        x = x.type(torch.FloatTensor).to(device)
        x_dot = x_dot.type(torch.FloatTensor).to(device)

        # one gradient step
        if model_type == "HyperSINDy":
            recon, kld = train_hyper(net, optim, x, x_dot, beta, weight_decay, device)
        elif model_type == "SINDy":
            recon, kld = train_sindy(net, optim, x, x_dot, weight_decay, device)

        recons += recon
        klds += kld
    return recons, klds 

def train_hyper(net, optim, x, x_dot, beta, weight_decay, device, clip):
    x_dot_pred, sindy_coeffs = net(x, device)
    recon = ((x_dot_pred - x_dot) ** 2).sum(1).mean()
    kld = net.kl(sindy_coeffs)
    masked_coeffs = sindy_coeffs * net.threshold_mask
    reg = (masked_coeffs ** 2).sum(1).sum(1).mean(0) * weight_decay
    loss = recon + kld * beta + reg
    optim.zero_grad()
    loss.backward()
    if clip is not None:
        nn.utils.clip_grad_norm_(net.parameters(), clip)
    optim.step()
    return recon.item(), kld.item()


def train_sindy(net, optim, x, x_dot, weight_decay, device, clip):
    x_dot_pred, sindy_coeffs = net(x, device)
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
    board.add_scalar("(x_dot_pred - x_dot) ** 2", recon, epoch)
    board.add_scalar("kld", kl, epoch)


def update_beta(beta, beta_increment, beta_max):
    beta += beta_increment
    if beta > beta_max:
        beta = beta_max
    return beta
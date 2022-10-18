import torch
import numpy as np
from tqdm import tqdm
from src.utils.other import save_model

def train(net, optim, scheduler, cp_path, model_type, trainloader, board,
          device, initial_epoch, epochs, beta_init, beta_increment,
          checkpoint_interval, threshold_timer, beta_max,
          weight_decay, noise_threshold, coef_threshold): 

    net.train()

    if beta_increment is None:
        beta_increment = beta_max / 100.0
    beta = 0
    thresh_determ = False
    for e in range(initial_epoch, epochs + initial_epoch):
        recons, klds = 0, 0
        for i, (x, x_dot) in enumerate(trainloader):
            x = x.type(torch.FloatTensor).to(device)
            x_dot = x_dot.type(torch.FloatTensor).to(device)

            # one gradient step
            if model_type == "HyperSINDy1":
                recon, kld = train_hyper1(net, optim, x, x_dot, beta, weight_decay, device)
            elif model_type == "HyperSINDy2":
                recon, kld = train_hyper2(net, optim, x, x_dot, beta, weight_decay, device)
            elif model_type == "HyperSINDy22":
                #recon, kld = train_hyper22(net, optim, x, x_dot, beta, weight_decay, device)
                recon, kld = train_hyper3_22_alternate(net, optim, x, x_dot, beta, weight_decay, device)
            elif model_type == "HyperSINDy3":
                #recon, kld = train_hyper3(net, optim, x, x_dot, beta, weight_decay, device)
                recon, kld = train_hyper3_22_alternate(net, optim, x, x_dot, beta, weight_decay, device)
            elif model_type == "SINDy":
                recon, kld = train_sindy(net, optim, x, x_dot, weight_decay, device)

            recons += recon
            klds += kld

        # log losses
        log_losses(board, recons / len(trainloader), klds / len(trainloader), e)

        # threshold
        if (e % threshold_timer == 0) and (e != 0) and (beta == beta_max):
            # threshold out every term except one in each equation (the max)
            threshold_more = e + threshold_timer * 4 >= epochs + initial_epoch
            # EM threshold - threshold the deterministic coefs or the noise coefs
            thresh_determ = not thresh_determ
            update_threshold_mask(net, model_type, device, coef_threshold,
                                  noise_threshold, threshold_more, thresh_determ)

        # save
        if (e + 1) % checkpoint_interval == 0:
            save_model(cp_path, net, optim, scheduler, e)

        scheduler.step()
        beta = update_beta(beta, beta_increment, beta_max)

    save_model(cp_path, net, optim, scheduler, e)

    return net, optim, scheduler

def update_threshold_mask(net, model_type, device, threshold1, threshold2,
                          threshold_more, thresh_determ):
    if model_type == 'SINDy':
        net.threshold_mask[torch.abs(net.sindy_coefficients) < threshold1] = 0
    elif model_type == 'HyperSINDy1':
        coefs = net.sindy_coeffs_full(batch_size=500, device=device)
        mean_coefs = torch.mean(coefs, dim=0)
        net.threshold_mask[torch.abs(mean_coefs) < threshold1] = 0
    elif model_type == 'HyperSINDy2':
        coefs = torch.mean(net.sindy_coeffs_full(batch_size=500, device=device), dim=0)
        net.threshold_mask[torch.abs(coefs) < threshold1] = 0
        net.threshold_mask[0] = 1
    elif model_type == 'HyperSINDy22':
        if thresh_determ:
            net.threshold_mask[torch.abs(net.sindy_coefficients) < threshold1] = 0
        else:
            noise_coefs = net.sample_transition(batch_size=500, device=device)
            #ab_coefs = torch.abs(torch.mean(noise_coefs, dim=0)) * net.threshold_mask_noise
            ab_coefs = torch.abs(torch.std(noise_coefs, dim=0)) * net.threshold_mask_noise
            max_val, max_idx = torch.max(ab_coefs, dim=0)
            if threshold_more:
                not_max = ~ab_coefs.eq(max_val)
                net.threshold_mask_noise[not_max] = 0
            else:
                net.threshold_mask_noise[ab_coefs < threshold2] = 0
                net.threshold_mask_noise[max_idx] = 1 # in case it thresholds out everything, we want at least 1 term in each equation
    elif model_type == 'HyperSINDy3':
        if thresh_determ:
            net.threshold_mask[torch.abs(net.sindy_coefficients) < threshold1] = 0
        else:
            noise_coefs = net.sample_transition(batch_size=500, device=device)
            #ab_coefs = torch.abs(torch.mean(noise_coefs, dim=0))
            ab_coefs = torch.abs(torch.std(noise_coefs, dim=0))
            net.threshold_mask_noise[ab_coefs < threshold2] = 0

# try a kind of expectation-maximization thresholding? i.e. alternating
# need term in the loss function for determinstica and term for stochastic
def train_hyper3_22_alternate(net, optim, x, x_dot, beta, weight_decay, device):
    x_dot_pred, x_dot_pred_theta, noise_theta, noise_coeffs = net(x, device)
    determ_error = x_dot - x_dot_pred_theta
    recon1 = (determ_error ** 2).sum(1).mean()
    recon2 = ((noise_theta - determ_error.detach().clone().to(device)) ** 2).sum(1).mean()
    recon = recon1 + recon2
    kld = net.kl(noise_coeffs)
    masked_coeffs = net.sindy_coefficients * net.threshold_mask
    reg1 = (masked_coeffs ** 2).sum() * weight_decay
    masked_coeffs = noise_coeffs * net.threshold_mask_noise
    reg2 = (masked_coeffs ** 2).sum(1).sum(1).mean() * weight_decay
    loss = recon + kld * beta + reg1 + reg2
    optim.zero_grad()
    loss.backward()
    optim.step()
    return recon.item(), kld.item()

# try a kind of expectation-maximization thresholding? i.e. alternating
# need term in the loss function for determinstica and term for stochastic
def train_hyper3(net, optim, x, x_dot, beta, weight_decay, device):
    x_dot_pred, x_dot_pred_theta, noise_theta, noise_coeffs = net(x, device)
    recon1 = ((x_dot_pred - x_dot) ** 2).sum(1).mean()
    recon2 = ((x_dot_pred_theta - x_dot) ** 2).sum(1).mean()
    recon = recon1 * 0.9 + recon2 * 0.1
    kld = net.kl(noise_coeffs)
    masked_coeffs = net.sindy_coefficients * net.threshold_mask
    reg1 = (masked_coeffs ** 2).sum() * weight_decay
    masked_coeffs = noise_coeffs * net.threshold_mask_noise
    reg2 = (masked_coeffs ** 2).sum(1).sum(1).mean() * weight_decay
    loss = recon + kld * beta + reg1 + reg2
    optim.zero_grad()
    loss.backward()
    optim.step()
    return recon.item(), kld.item()

# try a kind of expectation-maximization thresholding? i.e. alternating
# need term in the loss function for determinstica and term for stochastic
def train_hyper22_alternate(net, optim, x, x_dot, beta, weight_decay, device):
    x_dot_pred, x_dot_pred_theta, noise_theta, noise_coeffs = net(x, device)
    recon1 = ((x_dot_pred_theta - x_dot) ** 2).sum(1).mean()
    determ_error = (x_dot - x_dot_pred_theta).detach().clone().to(device)
    recon2 = ((noise_theta - determ_error) ** 2).sum(1).mean()
    recon = recon1 + recon2
    kld = net.kl(noise_coeffs)
    masked_coeffs = net.sindy_coefficients * net.threshold_mask
    reg1 = (masked_coeffs ** 2).sum() * weight_decay
    masked_coeffs = noise_coeffs * net.threshold_mask_noise
    reg2 = (masked_coeffs ** 2).sum(1).sum(1).mean() * weight_decay
    loss = recon + kld * beta + reg1 + reg2
    optim.zero_grad()
    loss.backward()
    optim.step()
    return recon.item(), kld.item()

def train_hyper2(net, optim, x, x_dot, beta, weight_decay, device):
    x_dot_pred, sindy_coeffs_full, sindy_coeffs = net(x, device)
    recon = ((x_dot_pred - x_dot) ** 2).sum(1).mean()
    kld = net.kl(sindy_coeffs)
    masked_coeffs = sindy_coeffs_full * net.threshold_mask
    reg = (masked_coeffs ** 2).sum(1).sum(1).mean(0) * weight_decay
    loss = recon + kld * beta + reg
    optim.zero_grad()
    loss.backward()
    optim.step()
    return recon.item(), kld.item()

def train_hyper1(net, optim, x, x_dot, beta, weight_decay, device):
    x_dot_pred, sindy_coeffs = net(x, device)
    recon = ((x_dot_pred - x_dot) ** 2).sum(1).mean()
    kld = net.kl(sindy_coeffs)
    masked_coeffs = sindy_coeffs * net.threshold_mask
    reg = (masked_coeffs ** 2).sum(1).sum(1).mean(0) * weight_decay
    loss = recon + kld * beta + reg
    optim.zero_grad()
    loss.backward()
    optim.step()
    return recon.item(), kld.item()

def train_sindy(net, optim, x, x_dot, weight_decay, device):
    x_dot_pred, sindy_coeffs = net(x, device)
    recon = ((x_dot_pred - x_dot) ** 2).sum(1).mean()
    masked_coeffs = sindy_coeffs * net.threshold_mask
    regularization = (masked_coeffs ** 2).sum() * weight_decay
    loss = recon + regularization
    optim.zero_grad()
    loss.backward()
    optim.step()
    return recon, 0


def log_losses(board, recon, kl, epoch):
    # tensorboard
    board.add_scalar("(x_dot_pred - x_dot) ** 2", recon, epoch)
    board.add_scalar("kld", kl, epoch)


def update_beta(beta, beta_increment, beta_max):
    beta += beta_increment
    if beta > beta_max:
        beta = beta_max
    return beta
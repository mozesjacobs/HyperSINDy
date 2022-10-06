import torch
import numpy as np
from tqdm import tqdm

def train(net, args, train_loader, train_board, optim, epoch, clip, beta, just_mean):
    net.train()

    # losses
    recon_losses = np.zeros([4])
    kl_loss = 0

    # for each batch
    for batch in tqdm(train_loader, desc="Training", total=len(train_loader), dynamic_ncols=True):
        (batch_recon, batch_kl), _ = net(batch, just_mean=just_mean)
        batch_loss = 0
        for i in range(len(batch_recon)):
            batch_loss += batch_recon[i]
            recon_losses[i] += batch_recon[i].item()
        batch_loss += batch_kl * beta
        kl_loss += batch_kl.item()

        # backprop
        optim.zero_grad()
        batch_loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        optim.step()

        if args.sequential_threshold is not None:
            #pass
            net.threshold_mask[torch.abs(net.sindy_coefficients) < net.sequential_threshold] = 0
            #net.threshold_mask[torch.abs(net.sindy_coefficients) > 1.0] = 1.0 / net.sindy_coefficients[torch.abs(net.sindy_coefficients) > 1.0].detach().clone().to(net.fc_mu.weight.device)


    log_losses(train_board, recon_losses / len(train_loader), kl_loss / len(train_loader), epoch)


def test(net, test_loader, test_board, epoch, timesteps, beta, just_mean):
    net.eval()
    recon_losses = np.zeros([4])
    kl_loss = 0
    for batch in tqdm(test_loader, desc="Testing", total=len(test_loader), dynamic_ncols=True):
        (batch_recon, batch_kl), _ = net(batch, just_mean=just_mean)
        for i in range(len(batch_recon)):
            recon_losses[i] += batch_recon[i].item()
        kl_loss += batch_kl.item()
    log_losses(test_board, recon_losses / len(test_loader), kl_loss / len(test_loader), epoch)


def log_losses(board, recon, kl, epoch):
    # tensorboard
    for i in range(len(recon)):
        board.add_scalar("x " + str(i + 1) + " recon", recon[i], epoch)
    board.add_scalar("kl", kl, epoch)
    #for i in range(len(kl)):
    #    board.add_scalar("kld " + str(i + 1), kl[i], epoch)
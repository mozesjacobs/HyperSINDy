import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from cmd_line import parse_args, hyperparameter_grid
from src.trainer.baseline import train
from src.utils.other import *
from src.utils.exp_utils import sample_trajectory, get_equations
from src.utils.plotting import plot_3d_trajectory, plot_1d_trajectory


def main():
    # get and save args
    args = parse_args()

    # train and val data (will refer to the val data as test data)
    train_set = load_data(args)

    # device
    torch.cuda.set_device(args.device)
    device = torch.cuda.current_device()

    # save args
    args_path, args_folder = get_args_path(args)
    make_folder(args_folder)
    save_args(args, args_path)

    # get hyperparameters for grid search
    settings = hyperparameter_grid()
    sampled_hyperparams = sample_hyperparameters(settings)

    # for each hyperparameter setting
    all_zs = []
    all_equations = []
    for i in range(settings['num_samples']):
        # get current hyperparams
        hyperparams = sampled_hyperparams[i]
        l1, l2, l3, l4 = hyperparams
        
        # checkpoint, experiments, tensorboard paths
        cp_path, cp_folder = get_checkpoint_path(args, hyperparams)
        exp_path_hps, exp_folder = get_experiments_path(args, hyperparams)
        tb_path, tb_folder = get_tb_path(args, hyperparams)
        make_folder(cp_folder)
        make_folder(exp_folder)
        print_folders(args.print_folder, cp_folder, exp_folder, tb_folder)

        # boards
        train_board = SummaryWriter(tb_path, purge_step=True)

        # create model, optim, scheduler, initial epoch
        net, optim, scheduler, initial_e = make_model(args, device)
        
        # load model, optim, scheduler, epoch from checkpoint
        if args.load_cp == 1:
            net, optim, scheduler, initial_e = load_checkpoint(cp_path, net, optim, scheduler, device)

        # dataloader
        trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)

        # train model
        train(net, optim, scheduler, cp_path, args.model, trainloader, train_board,
              args.device, initial_e, args.epochs, args.beta_init, args.beta_inc, args.threshold_interval,
              args.checkpoint_interval,
              beta_max=l1, weight_decay=l2, noise_threshold=l3, coef_threshold=l4)
              
        # sample trajectory
        zs = sample_trajectory(net, device, train_set.x[0].numpy(), args.exp_batch_size, args.exp_dt, args.exp_timesteps)

        # get equations
        equations = get_equations(net, args.model, device, args.statistic_batch_size,
                                  args.z_dim, args.poly_order,
                                  args.include_constant, args.include_sine)
        
        all_zs.append(zs)
        all_equations.append(equations)
    
    all_zs = np.stack(all_zs, axis=0)
    if args.z_dim == 3:
        plot_3d_trajectory(exp_folder + "comparison.png", all_zs, sampled_hyperparams, all_equations, figsize=None)
    elif args.z_dim == 1:
        plot_1d_trajectory(exp_folder + "comparison.png", train_set.x.numpy(), all_zs, sampled_hyperparams, all_equations, figsize=None)

if __name__ == "__main__":
    main()
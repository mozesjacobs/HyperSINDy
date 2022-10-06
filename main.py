import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import json
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from cmd_line import parse_args
from src.trainer.baseline import train, test
from src.utils.other import *
from src.utils.model_utils import init_weights


def main():
    # get and save args
    args = parse_args()

    # train and val data (will refer to the val data as test data)
    train_set, test_set, _ = load_data(args)

    # dataloaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=1)

    # boards
    train_name, test_name = get_tb_path(args)
    train_board = SummaryWriter(train_name, purge_step=True)
    test_board = SummaryWriter(test_name, purge_step=True)

    print(args.device)

    # device
    torch.cuda.set_device(args.device)
    device = torch.cuda.current_device()

    # checkpoint, args, experiments path
    cp_path, cp_folder = get_checkpoint_path(args)
    args_path, args_folder = get_args_path(args)
    exp_folder = get_experiments_path(args)
    make_folder(cp_folder)
    make_folder(exp_folder)
    print_folders(args.print_folder, cp_folder, exp_folder, train_name[:-5])

    # save args
    with open(args_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # create model, optim, scheduler, initial epoch
    net = make_model(args).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.adam_regularization)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=args.gamma_factor)
    initial_e = 0
    
    # load model, optim, scheduler, epoch from checkpoint
    if args.load_cp == 1:
        net, optim, scheduler, initial_e = load_model(cp_path, device, net, optim, scheduler)
    else:  # init network
        net.apply(init_weights)

    #print(net.sindy_coefficients)

    # for each epoch
    for epoch in tqdm(range(args.epochs), desc="Epoch", total=args.epochs, dynamic_ncols=True):
        # train
        train(net, args, train_loader, train_board, optim, epoch + initial_e, args.clip, args.beta, args.just_mean)

        # test
        if (epoch + 1) % args.test_interval == 0:
            test(net, test_loader, test_board, epoch + initial_e, args.timesteps, args.beta, args.just_mean)
        
        # step on learning rate scheduler
        scheduler.step()
    
        # save checkpoints
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint = {'epoch': epoch + initial_e,
                          'model': net.state_dict(),
                          'optimizer': optim.state_dict(),
                          'scheduler': scheduler.state_dict()}
            torch.save(checkpoint, cp_path)
    
    #print(net.sindy_coefficients)

if __name__ == "__main__":
    main()
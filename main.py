import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from cmd_line import parse_hyperparams, parse_args
from src.trainer.baseline import train
from src.utils.other import *
from src.utils.exp_utils import sample_trajectory, get_equations
from src.utils.plotting import plot_trajectory


def main():
    # get and save args
    args = parse_args()
    hyperparams = parse_hyperparams()

    # train and val data (will refer to the val data as test data)
    train_set = load_data(args)

    # device
    torch.cuda.set_device(args.device)
    device = torch.cuda.current_device()

    # args, checkpoint, tensorboard paths
    args_path, args_folder = get_args_path(args, hyperparams)
    cp_path, cp_folder = get_checkpoint_path(args, hyperparams)
    tb_path, tb_folder = get_tb_path(args, hyperparams)
    make_folder(args_folder)
    make_folder(cp_folder)
    save_args(args, args_path)
    print_folders(args.print_folder, cp_folder, tb_folder)

    # boards
    train_board = SummaryWriter(tb_path, purge_step=True)
    #train_board.add_hparams(hparam_dict=vars(hyperparams), metric_dict={},
    #train_board.add_hparams(hparam_dict=vars(hyperparams),
    #    metric_dict={'hparam/accuracy': 25, 'hparam/accuracy2': 25},
    #    hparam_domain_discrete=None, run_name='')

    # create model, optim, scheduler, initial epoch
    net, optim, scheduler, initial_epoch = make_model(args, hyperparams, device)
    
    # load model, optim, scheduler, epoch from checkpoint
    if args.load_cp == 1:
        net, optim, scheduler, initial_epoch = load_checkpoint(
            cp_path, net,
            optim, scheduler, device)

    # dataloader
    trainloader = DataLoader(train_set, batch_size=hyperparams.batch_size,
                             shuffle=True, num_workers=1, drop_last=True)

    # train model
    train(net, args, hyperparams, optim, scheduler, trainloader, train_set, 
          train_board, cp_path, initial_epoch, device)

    # close
    train_board.flush()
    train_board.close()

if __name__ == "__main__":
    main()
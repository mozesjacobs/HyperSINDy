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
from src.utils.other import *#load_data, load_model, make_model, get_tb_path, get_checkpoint_path, get_args_path, get_experiments_path
from src.utils.model_utils import init_weights
from src.utils.exp_utils import *


def main():
    # get and save args
    args = parse_args()

    # train and val data (will refer to the val data as test data)
    _, test_set, _ = load_data(args)

    # dataloaders
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=1)

    # device
    torch.cuda.set_device(args.device)
    device = torch.cuda.current_device()

    # checkpoint, args, experiments path
    cp_path, cp_folder = get_checkpoint_path(args)
    args_path, args_folder = get_args_path(args)
    exp_folder = get_experiments_path(args)
    make_folder(cp_folder)
    make_folder(exp_folder)
    print_folders(args.print_folder, cp_folder, None)

    # load model
    net = make_model(args).to(device)
    net, _, _, _ = load_model(cp_path, device, net)

    # create settings name of experiment
    exp_ext = 'ejm-' + str(args.exp_just_mean)
    exp_ext += '-zs-' + str(args.z_plot_start)
    exp_ext += '-ze-' + str(args.z_plot_end)
    exp_ext += '-didx-' + str(args.data_idx)
    exp_ext += '-v1-' + str(args.v1)
    exp_ext += '-v2-' + str(args.v2)
    if args.e9 == 1 or args.run_all == 1:
        exp_ext += '-erpo-' + str(args.exp_return_post)
    if len(args.custom_plot_name) != 0:
        exp_ext += '-' + args.custom_plot_name
    #print(args.exp_just_mean)
 
    # gif of predicted high-dim lorenz
    if args.e1 == 1 or args.run_all == 1:
        exp_folder = exp_folder + "exp1/"
        make_folder(exp_folder)
        exp1(args, net, exp_folder +  exp_ext + '.gif')

    # trajectory of latents
    if args.e2 == 1 or args.run_all == 1:
        exp_folder = exp_folder + "exp2/"
        make_folder(exp_folder)
        exp2(args, net, exp_folder + exp_ext + '.png')

    # unconditioned trajectory of latents
    if args.e3 == 1 or args.run_all == 1:
        exp_folder = exp_folder + "exp3/"
        make_folder(exp_folder)
        exp3(args, net, exp_folder + exp_ext + '.png')

    # sample frmo prior
    if args.e4 == 1 or args.run_all == 1:
        exp_folder = exp_folder + "exp4/"
        make_folder(exp_folder)
        exp4(args, net, exp_folder + exp_ext + '.png')

    # latent vector hankle trajectory plot
    if args.e5 == 1 or args.run_all == 1:
        exp_folder = exp_folder + "exp5/"
        make_folder(exp_folder)
        exp5(args, net, exp_folder + exp_ext + '.png')

    # latent vector hankle trajectory plot using initial x
    if args.e6 == 1 or args.run_all == 1:
        exp_folder = exp_folder + "exp6/"
        make_folder(exp_folder)
        exp6(args, net, exp_folder + exp_ext + '.png')

    # use true z and sample forward
    if args.e7 == 1 or args.run_all == 1:
        exp_folder = exp_folder + "exp7/"
        make_folder(exp_folder)
        exp7(args, net, exp_folder + exp_ext + '.png')

    # predict latent trajectory and plot it for a z, not delay
    if args.e8 == 1 or args.run_all == 1:
        exp_folder = exp_folder + "exp8/"
        make_folder(exp_folder)
        exp8(args, net, exp_folder + exp_ext + '.png')

    # trajectory of latents when true latent is given
    if args.e9 == 1 or args.run_all == 1:
        exp_folder = exp_folder + "exp9/"
        make_folder(exp_folder)
        exp9(args, net, exp_folder + exp_ext + '.png')

    if args.e10 == 1 or args.run_all == 1:
        exp_folder = exp_folder + "exp10/"
        make_folder(exp_folder)
        exp10(args, net, exp_folder + exp_ext + '.png')

    if args.e11 == 1 or args.run_all == 1:
        exp_folder = exp_folder + "exp11/"
        make_folder(exp_folder)
        exp11(args, net, exp_folder + exp_ext + '.png')
    
    if args.e12 == 1 or args.run_all == 1:
        exp_folder = exp_folder + "exp12/"
        make_folder(exp_folder)
        exp12(args, net, exp_folder + exp_ext + '.png')

    if args.e13 == 1 or args.run_all == 1:
        exp_folder = exp_folder + "exp13/"
        make_folder(exp_folder)
        exp13(args, net, exp_folder + exp_ext + '.gif')

    if args.e14 == 1 or args.run_all == 1:
        exp_folder = exp_folder + "exp14/"
        make_folder(exp_folder)
        exp14(args, net, exp_folder + exp_ext + '.png')

    # print the folder
    print_folders(args.print_folder, None, exp_folder)

    """
    python3 main.py && python3 experiments.py --e6 1 && python3 experiments.py --e6 1 --z_plot_start 5 && python3 experiments.py --e6 1 --z_plot_start 5 --v1 60 --v2 45 && python3 experiments.py --e6 1 --exp_just_mean False && python3 experiments.py --e6 1 --exp_just_mean False --z_plot_start 5 && python3 experiments.py --e6 1 --exp_just_mean False --z_plot_start 5 --v1 60 --v2 45
    python3 main.py && python3 experiments.py --e6 1 && python3 experiments.py --e6 1 --z_plot_start 5 && python3 experiments.py --e6 1 --z_plot_start 5 --v1 60 --v2 45
    python3 main.py && python3 experiments.py --e6 1 --exp_just_mean False && python3 experiments.py --e6 1 --exp_just_mean False --z_plot_start 5 && python3 experiments.py --e6 1 --exp_just_mean False --z_plot_start 5 --v1 60 --v2 45
    python3 experiments.py --e6 1 && python3 experiments.py --e6 1 --z_plot_start 5 && python3 experiments.py --e6 1 --z_plot_start 5 --v1 60 --v2 45
    python3 experiments.py --e6 1 --exp_just_mean False && python3 experiments.py --e6 1 --exp_just_mean False --z_plot_start 5 && python3 experiments.py --e6 1 --exp_just_mean False --z_plot_start 5 --v1 60 --v2 45
    """


if __name__ == "__main__":
    main()
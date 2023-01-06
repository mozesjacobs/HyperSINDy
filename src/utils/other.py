import os
import json
import torch
from src.dataset.Datasets import SyntheticDataset
from src.utils.path_utils import *
from src.utils.model_utils import init_weights


def load_data(args, hyperparams):
    """Loads the data.

    Loads the data using the given arguments and hyperparameters.

    Args:
        args: The argparser object return by parse_args() in the file
            cmd_line.py.
        hyperparams: The argparser object returned by parse_hyperparams() in 
            the file cmd_line.py

    Returns:
        A SyntheticDataset containing the data.
    """
    # train and val data (using val as "test" data)
    fpath = get_data_path(args.data_folder, args.dataset,
                          args.noise_type, str(args.noise_scale))
    return SyntheticDataset(args, hyperparams, fpath)

def load_checkpoint(cp_path, net, optim, scheduler, device):
    """Loads the last checkpoint.

    Loads the latest checkpoint at cp_path into the latest epoch and the given
    network, optimizer, and scheduler.

    Args:
        cp_path: The string (relative) path to the checkpoint to load.
        net: The network (nn.Module) to load into.
        optim: The optimizer (torch.optim) to load into.
        scheduler: The torch.optim.lr_scheduler to load into.
        device: The cpu or gpu device to load the checkpoint (and network)
            onto. For cpu, device must be "cpu". For gpu, the device must be
            an integer corresponding to the gpu to be used (i.e.: 0 or 1 or 2
            or 3).

    Returns:
        A tuple (Net, Optim, Scheduler, Initial_e). Net is the nn.Module that
        was loaded from the checkpoint. Optim is the torch.optim that was
        loaded from the checkpoint. Scheduler is the torch.optim.lr_scheduler
        that was loaded from the checkpoint. Initial_e is an integer describing
        which epoch in training was loaded from the checkpoint.
    """
    checkpoint = torch.load(cp_path, map_location="cuda:" + str(device))
    net.load_state_dict(checkpoint['model'])
    net.to(device)
    optim.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    initial_e = checkpoint['epoch']
    return net, optim, scheduler, initial_e

def make_model(args, hyperparams, device):
    """Construct a model.

    Creates a HyperSINDy, ESINDy, or SINDy model, as well as an optimizer,
    learning rate scheduler, and initial epoch, using the given parameters.

    Args:
        args: The argparser object return by parse_args() in the file
            cmd_line.py.
        hyperparams: The argparser object returned by parse_hyperparams() in 
            the file cmd_line.py
        device: The cpu or gpu device to construct the network with. To use
            cpu, device must be "cpu". To use a gpu, specify, the device number
            of the gpu as an integer (i.e.: 0 or 1 or 2 or 3). 

    Returns:
        A tuple (Net, Optim, Scheduler, Initial_e). Net is the nn.Module that
        was created. Optim is the torch.optim that was created. Scheduler is
        the torch.optim.lr_scheduler that was created. Initial_e is an integer
        describing the starting epoch of training (0).
    """
    if args.model == 'HyperSINDy':
        from src.models.HyperSINDy import Net
    elif args.model == 'ESINDy':
        from src.models.ESINDy import Net
    elif args.model == 'SINDy':
        from src.models.SINDy import Net
    net = Net(args, hyperparams).to(device)
    net.apply(init_weights)
    optim = torch.optim.Adam(
        net.parameters(), lr=hyperparams.learning_rate,
        weight_decay=hyperparams.adam_reg)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optim, gamma=args.gamma_factor)
    return net, optim, scheduler, 0

def make_folder(folder):
    """Creates a folder.

    Creates the given folder, unless it already exists.

    Args:
        folder: A str denoting the path to the folder to create.
    
    Returns:
        None
    """
    if not os.path.isdir(folder):
        os.system("mkdir -p " + folder)

def print_folders(do_print, cp_folder=None, tb_folder=None):
    """Prints the folders.

    Prints the given folders.

    Args:
        do_print: A boolean denoting whether to print the folders to terminal.
            If False, will not print.
        cp_folder: The str path to the folder where the model checkpoints are
            saved during training. If None, does not print.
        tn_folder: The str path to the folder where the tensorboard logs for
            the current model are saved. If None, does not print.

    Returns:
        None
    """
    if do_print:
        if cp_folder is not None:
            print("Checkpoints saved at:        ", cp_folder)
        if tb_folder is not None:
            print("Tensorboard logs saved at:   ", tb_folder)

def save_model(cp_path, net, optim, scheduler, epoch):
    """Saves the current checkpoint.

    Saves the current network, optimizer, scheduler, and epoch as a .pt file
    at the specified path. Overwrites the file if one exists at the given path.

    Args:
        cp_path: The string (relative) path to the checkpoint to save.
        net: The network (nn.Module) to save.
        optim: The optimizer (torch.optim) to save.
        scheduler: The torch.optim.lr_scheduler to save.
        epoch: The current epoch in training to save.

    Returns:
        None
    """
    checkpoint = {'epoch': epoch,
                  'model': net.state_dict(),
                  'optimizer': optim.state_dict(),
                  'scheduler': scheduler.state_dict()}
    torch.save(checkpoint, cp_path)

def save_args(args, args_path):
    """Saves the arguments.

    Saves the given args as a json file at the given path. Overwrites the file
    at the given path if it already exists.

    Args:
        args: The argparser object return by parse_args() in the file
            cmd_line.py.
        args_path: The path to the json file to save args as.

    Returns:
        None
    """
    with open(args_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
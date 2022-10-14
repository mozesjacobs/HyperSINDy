import os
import json
from src.dataset.Datasets import *
from src.utils.path_utils import *
from src.utils.model_utils import init_weights


def load_data(args):
    # train and val data (using val as "test" data)
    if args.data_set == "lorenz":
        fpath = get_lorenz_path(args.noise_type, args.noise_scale)
        trainset = LorenzDataset(fpath)
    elif args.data_set == "pupil":
        fpath = get_pupil_path()
        trainset = PupilDataset(datapath=fpath, norm=args.norm_data, scale=args.scale_data, amount=None)
    return trainset

def load_model(cp_path, net, optim, scheduler, device):
    checkpoint = torch.load(cp_path, map_location="cuda:" + str(device))
    net.load_state_dict(checkpoint['model'])
    net.to(device)
    optim.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    initial_e = checkpoint['epoch']
    return net, optim, scheduler, initial_e

def make_model(args, device):
    if args.model == 'HyperSINDy1':
        from src.models.HyperSINDy1 import Net
    elif args.model == 'HyperSINDy2':
        from src.models.HyperSINDy2 import Net
    elif args.model == 'HyperSINDy22':
        from src.models.HyperSINDy22 import Net
    elif args.model == 'HyperSINDy3':
        from src.models.HyperSINDy3 import Net
    elif args.model == 'SINDy':
        from src.models.SINDy import Net
    net = Net(args).to(device)
    net.apply(init_weights)
    optim = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.adam_regularization)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=args.gamma_factor)
    return net, optim, scheduler, 0

def make_folder(folder):
    if not os.path.isdir(folder):
        os.system("mkdir -p " + folder)

def print_folders(do_print, cp_folder=None, exp_folder=None, tb_folder=None):
    if do_print:
        if cp_folder is not None:
            print("Checkpoints saved at:        ", cp_folder)
        if exp_folder is not None:
            print("Experiment results saved at: ", exp_folder)
        if tb_folder is not None:
            print("Tensorboard logs saved at:   ", tb_folder)
        
def load_model(cp_path, device, net, optim=None, scheduler=None):
    checkpoint = torch.load(cp_path, map_location="cuda:" + str(device))
    net.load_state_dict(checkpoint['model'])
    net.to(device)
    if optim is not None:
        optim.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    return net, optim, scheduler

def save_model(cp_path, net, optim, scheduler, epoch):
    checkpoint = {'epoch': epoch,
                  'model': net.state_dict(),
                  'optimizer': optim.state_dict(),
                  'scheduler': scheduler.state_dict()}
    torch.save(checkpoint, cp_path)

def save_args(args, args_path):
    with open(args_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

# (beta, weight_decay, noise_reg, sindy_reg)
def sample_hyperparameters(settings):
    samples = []
    for i in range(settings['num_samples']):
        resample = True
        while resample:
            l1 = np.random.choice(settings['beta'])
            l2 = np.random.choice(settings['sindy_weight_decay'])
            l3 = np.random.choice(settings['noise_coef_threshold'])
            l4 = np.random.choice(settings['sindy_coef_threshold'])
            params = (l1, l2, l3, l4)
            if params not in samples:
                samples.append(params)
                resample=False
    return samples
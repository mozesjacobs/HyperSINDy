import os

def get_lorenz_path(noise_type, noise_scale):
    return "data/lorenz/state-" + noise_type + "_scale-" + str(noise_scale) + "/"

def get_lorenz2_path(noise_type, noise_scale):
    return "data/lorenz2/state-" + noise_type + "_scale-" + str(noise_scale) + "/"

def get_lorenz3_path(noise_type, noise_scale):
    return "data/lorenz3/state-" + noise_type + "_scale-" + str(noise_scale) + "/"

def get_pupil_path():
    return "data/pupil1.npy"

def get_hyperparam_path(hyperparams):
    path = "l1-" + str(hyperparams[0]) + "_l2-" + str(hyperparams[1])
    return path + "_l2-" + str(hyperparams[2]) + "_l3-" + str(hyperparams[3]) + "/"

def get_general_path(args):
    path = args.data_set + "/state-" + args.noise_type + "_scale-" + str(args.noise_scale) + "/"
    if args.session_name is not None and args.session_name != '':
        path += args.model + "/" + args.date + "/" + args.session_name + "/"
    else:
        path += args.model + "/" + args.date + "/"
    return path

def get_checkpoint_path(args, hyperparams):
    cp_folder = args.model_folder + get_general_path(args)
    cp_folder += get_hyperparam_path(hyperparams)
    return cp_folder + "checkpoint.pt", cp_folder

def get_args_path(args):
    args_folder = args.model_folder + get_general_path(args)
    return args_folder + "args.txt", args_folder

def get_tb_path(args, hyperparams):
    tb_folder = args.tensorboard_folder + get_general_path(args)
    tb_folder += get_hyperparam_path(hyperparams)
    return tb_folder + "train", tb_folder

def get_experiments_path(args, hyperparams):
    exp_folder = args.experiments + get_general_path(args)
    exp_folder_hps = exp_folder + get_hyperparam_path(hyperparams)
    return exp_folder_hps, exp_folder
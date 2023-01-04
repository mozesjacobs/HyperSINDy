import os

def get_sub_data_path(dataset, noise_type, noise_scale):
    return dataset + "/state-" + noise_type + "_scale-" + str(noise_scale) + "/"

def get_data_path(data_folder, dataset, noise_type, noise_scale):
    return data_folder + get_sub_data_path(dataset, noise_type, noise_scale)

def get_hparams_path(hpd):
    path = "B-" + str(hpd.beta) + "_WD-" + str(hpd.weight_decay) + "_T-" + str(hpd.threshold)
    path += "_TI-" + str(hpd.threshold_interval) + "_ND-" + str(hpd.threshold_interval)
    path += "_HD-" + str(hpd.hidden_dim) + "_E-" + str(hpd.epochs)
    path += "_LR-" + str(hpd.learning_rate) + "_BS-" + str(hpd.batch_size)
    path += "_AR-" + str(hpd.adam_reg) + "_C-" + str(hpd.clip)
    path += "_PR-" + str(hpd.prior)
    return path

def get_general_path(args, hparams):
    path = get_sub_data_path(args.dataset, args.noise_type, args.noise_scale)
    path += args.model + "/" + args.date + "/" + args.session_name + "/"
    return path + get_hparams_path(hparams) + "/"

def get_checkpoint_path(args, hparams):
    cp_folder = args.model_folder + get_general_path(args, hparams)
    return cp_folder + "checkpoint.pt", cp_folder

def get_args_path(args, hparams):
    args_folder = args.model_folder + get_general_path(args, hparams)
    return args_folder + "args.txt", args_folder

def get_tb_path(args, hparams):
    tb_folder = args.tensorboard_folder + get_general_path(args, hparams)
    return tb_folder + "train", tb_folder
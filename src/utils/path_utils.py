import os

def get_sub_data_path(dataset, noise_type, noise_scale):
    """Gets a part of the path to the data.
    
    Gets a component of the path to the data files the sub_data_path. This path
    includes the dataset, the noise type, and the noise scale.

    Args:
        dataset: The dataset name as a string.
        noise_type: The type of noise used in the data as a string.
        noise_scale: The scale of the noise used in the data, as a float.

    Returns:
        A string of the path. 
    """
    return dataset + "/state-" + noise_type + "_scale-" + str(noise_scale) + "/"

def get_data_path(data_folder, dataset, noise_type, noise_scale):
    """Gets the full path to the data.

    Gets the full path to the data, excluding the actual file name of the data.

    Args:
        data_folder: The base folder (str) where all the data is stored.
        dataset: The dataset name as a string.
        noise_type: The type of noise used in the data as a string.
        noise_scale: The scale of the noise used in the data, as a float.

    Returns:
        A string of the path.
    """
    return data_folder + get_sub_data_path(dataset, noise_type, noise_scale)

def get_hparams_path(hpd):
    """Gets the hyperparameters as a path.

    Converts the hyperparameters into a part of a file path.

    Args:
        hpd: The argparser object returned by parse_hyperparams() in 
            the file cmd_line.py. Denotes the hyperparameters.
    
    Returns:
        A string of the path.
    """
    path = "B-" + str(hpd.beta) + "_WD-" + str(hpd.weight_decay)
    path += "_ST-" + str(hpd.soft_threshold) + "_HT-" + str(hpd.hard_threshold)
    path += "_TI-" + str(hpd.threshold_interval) + "_ND-" + str(hpd.noise_dim)
    path += "_HD-" + str(hpd.hidden_dim) + "_E-" + str(hpd.epochs)
    path += "_LR-" + str(hpd.learning_rate) + "_BS-" + str(hpd.batch_size)
    path += "_AR-" + str(hpd.adam_reg) + "_C-" + str(hpd.clip)
    path += "_PR-" + str(hpd.prior) + "_NE-" + str(hpd.num_ensemble)
    return path

def get_general_path(args, hparams):
    """Gets a general file path.

    Converts the args and hyperparameters into a file path.

    Args:
        args: The argparser object return by parse_args() in the file
            cmd_line.py.
        hparams: The argparser object returned by parse_hyperparams() in the
            ile cmd_line.py.
    
    Returns:
        A string of the path.
    """
    path = get_sub_data_path(args.dataset, args.noise_type, args.noise_scale)
    path += args.model + "/" + args.date + "/" + args.session_name + "/"
    return path + get_hparams_path(hparams) + "/"

def get_checkpoint_path(args, hparams):
    """Gets the path to the checkpoint.

    Gets the full path to where the model checkpoints are saved for the run
    specified by the given arguments.

    Args:
        args: The argparser object return by parse_args() in the file
            cmd_line.py.
        hparams: The argparser object returned by parse_hyperparams() in the
            ile cmd_line.py.
    
    Returns:
        A tuple of (string_a, string_b), where
        string_a = string_b + "checkpoint.pt" and string_b is the full path to
        the folder where the checkpoints are saved.
    """
    cp_folder = args.model_folder + get_general_path(args, hparams)
    return cp_folder + "checkpoint.pt", cp_folder

def get_args_path(args, hparams):
    """Gets the path to the arguments file.

    Gets the full path to where the arguments are saved for the run specified
    by the given arguments.

    Args:
        args: The argparser object return by parse_args() in the file
            cmd_line.py.
        hparams: The argparser object returned by parse_hyperparams() in the
            ile cmd_line.py.
    
    Returns:
        A tuple of (string_a, string_b), where string_a = string_b + "args.txt"
        and string_b is the full path to the folder where the arguments are
        saved.
    """
    args_folder = args.model_folder + get_general_path(args, hparams)
    return args_folder + "args.txt", args_folder

def get_tb_path(args, hparams):
    """Gets the path to the SummaryWriter.

    Gets the full path to where the SummaryWriter logs results for the run
    specified by the given arguments.

    Args:
        args: The argparser object return by parse_args() in the file
            cmd_line.py.
        hparams: The argparser object returned by parse_hyperparams() in the
            ile cmd_line.py.
    
    Returns:
        A tuple of (string_a, string_b), where string_a = string_b + "train/"
        and string_b is the full path to the folder where string_a is
        contained.
    """
    tb_folder = args.tensorboard_folder + get_general_path(args, hparams)
    return tb_folder + "train", tb_folder
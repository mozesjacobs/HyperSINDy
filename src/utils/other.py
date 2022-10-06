import os
from src.dataset.Datasets import *

def load_data(args):
    # train and val data (using val as "test" data)
    if args.data_set == "lorenz":
        folder, data_paths = get_lorenz_path()
        train_set = LorenzDataset(args, data_paths[0])
        val_set = LorenzDataset(args, data_paths[1])
        test_set = LorenzDataset(args, data_paths[2])
    elif args.data_set == "lorenz2":
        folder, data_paths = get_lorenz2_path()
        train_set = Lorenz2Dataset(args, data_paths[0])
        val_set = Lorenz2Dataset(args, data_paths[1])
        test_set = Lorenz2Dataset(args, data_paths[2])
    elif args.data_set == "lorenz3":
        folder, data_paths = get_lorenz3_path()
        train_set = Lorenz3Dataset(args, data_paths[0])
        val_set = Lorenz3Dataset(args, data_paths[1])
        test_set = Lorenz3Dataset(args, data_paths[2])
    elif args.data_set == "lorenz4":
        folder, data_paths = get_lorenz4_path()
        train_set = Lorenz4Dataset(args, data_paths[0])
        val_set = Lorenz4Dataset(args, data_paths[1])
        test_set = Lorenz4Dataset(args, data_paths[2])
    elif args.data_set == "lorenz5":
        folder, data_paths = get_lorenz5_path()
        train_set = Lorenz5Dataset(args, data_paths[0])
        val_set = Lorenz5Dataset(args, data_paths[1])
        test_set = Lorenz5Dataset(args, data_paths[2])
    elif args.data_set == "lorenz_simple":
        folder, data_paths = get_delay_lorenz_path()
        train_set = LorenzSimpleDataset(args, data_paths[0])
        val_set = LorenzSimpleDataset(args, data_paths[1], train_set.z_mean, train_set.z_std)
        test_set = LorenzSimpleDataset(args, data_paths[2], train_set.z_mean, train_set.z_std)
    elif args.data_set == "lorenz_simple_derivative":
        folder, data_paths = get_delay_lorenz_path()
        train_set = LorenzSimpleDerivativeDataset(args, data_paths[0])
        val_set = LorenzSimpleDerivativeDataset(args, data_paths[1])
        test_set = LorenzSimpleDerivativeDataset(args, data_paths[2])
    elif args.data_set == "lorenz_noise_simple":
        folder, data_paths = get_delay_lorenz_noise_path()
        train_set = LorenzSimpleDataset(args, data_paths[0])
        val_set = LorenzSimpleDataset(args, data_paths[1])
        test_set = LorenzSimpleDataset(args, data_paths[2])
    elif args.data_set == "delay_lorenz":
        folder, data_paths = get_delay_lorenz_path()
        train_set = DelayLorenzDataset(args, data_paths[0])
        val_set = DelayLorenzDataset(args, data_paths[1],
                                     train_set.U_x1, train_set.U_x2, train_set.U_z,
                                     train_set.x1_mean, train_set.x2_mean, train_set.z_mean,
                                     train_set.x1_std, train_set.x2_std, train_set.z_std)
        #val_set = train_set
        test_set = DelayLorenzDataset(args, data_paths[2],
                                      train_set.U_x1, train_set.U_x2, train_set.U_z,
                                      train_set.x1_mean, train_set.x2_mean, train_set.z_mean,
                                      train_set.x1_std, train_set.x2_std, train_set.z_std)
    elif args.data_set == "delay_lorenz_flex":
        folder, data_paths = get_delay_lorenz_path()
        train_set = DelayLorenzFlexDataset(args, data_paths[0])
        val_set = DelayLorenzFlexDataset(args, data_paths[1],
                                         train_set.U_x1, train_set.U_x2, train_set.U_z,
                                         train_set.x1_mean, train_set.x2_mean, train_set.z_mean,
                                         train_set.x1_std, train_set.x2_std, train_set.z_std)
        #val_set = train_set
        test_set = DelayLorenzFlexDataset(args, data_paths[2],
                                          train_set.U_x1, train_set.U_x2, train_set.U_z,
                                          train_set.x1_mean, train_set.x2_mean, train_set.z_mean,
                                          train_set.x1_std, train_set.x2_std, train_set.z_std)
    elif args.data_set == "hankel_lorenz":
        folder, data_paths = get_delay_lorenz_path()
        train_set = HankelLorenzDataset(args, data_paths[0])
        val_set = HankelLorenzDataset(args, data_paths[1],
                                     train_set.x1_mean, train_set.x2_mean, train_set.z_mean,
                                     train_set.x1_std, train_set.x2_std, train_set.z_std)
        #val_set = train_set
        test_set = HankelLorenzDataset(args, data_paths[2],
                                      train_set.x1_mean, train_set.x2_mean, train_set.z_mean,
                                      train_set.x1_std, train_set.x2_std, train_set.z_std)
    elif args.data_set == "stochastic_lorenz":
        folder, data_paths = get_stochastic_lorenz_path()
        train_set = StochasticLorenzDataset(args, data_paths[0])
        val_set = StochasticLorenzDataset(args, data_paths[1],
                                     train_set.x1_mean, train_set.x2_mean, train_set.z_mean,
                                     train_set.x1_std, train_set.x2_std, train_set.z_std)
        #val_set = train_set
        test_set = StochasticLorenzDataset(args, data_paths[2],
                                      train_set.x1_mean, train_set.x2_mean, train_set.z_mean,
                                      train_set.x1_std, train_set.x2_std, train_set.z_std)
    elif args.data_set == "stochastic_lorenz_flex":
        folder, data_paths = get_stochastic_lorenz_path()
        train_set = StochasticLorenzFlexDataset(args, data_paths[0])
        val_set = StochasticLorenzFlexDataset(args, data_paths[1],
                                     train_set.x1_mean, train_set.x2_mean, train_set.z_mean,
                                     train_set.x1_std, train_set.x2_std, train_set.z_std)
        #val_set = train_set
        test_set = StochasticLorenzFlexDataset(args, data_paths[2],
                                      train_set.x1_mean, train_set.x2_mean, train_set.z_mean,
                                      train_set.x1_std, train_set.x2_std, train_set.z_std)
    elif args.data_set == "delay_lorenz_simple":
        folder, data_paths = get_delay_lorenz_path()
        train_set = DelayLorenzSimpleDataset(args, data_paths[0])
        val_set = DelayLorenzSimpleDataset(args, data_paths[1], train_set.U_z, train_set.norm_mean, train_set.norm_std)
        #val_set = train_set
        test_set = DelayLorenzSimpleDataset(args, data_paths[2], train_set.U_z, train_set.norm_mean, train_set.norm_std)
    elif args.data_set == "delay_lorenz_noise_simple":
        folder, data_paths = get_delay_lorenz_noise_path()
        train_set = DelayLorenzSimpleDataset(args, data_paths[0])
        val_set = DelayLorenzSimpleDataset(args, data_paths[1], train_set.U_z)
        #val_set = train_set
        test_set = DelayLorenzSimpleDataset(args, data_paths[2], train_set.U_z)
    elif args.data_set == "delay_lorenz_big":
        folder, data_paths = get_delay_lorenz_big_path()
        train_set = DelayLorenzDataset(args, data_paths[0])
        val_set = DelayLorenzDataset(args, data_paths[1],
                                     train_set.U_x1, train_set.U_x2, train_set.U_z,
                                     train_set.x1_mean, train_set.x2_mean, train_set.z_mean,
                                     train_set.x1_std, train_set.x2_std, train_set.z_std)
        #val_set = train_set
        test_set = DelayLorenzDataset(args, data_paths[2],
                                      train_set.U_x1, train_set.U_x2, train_set.U_z,
                                      train_set.x1_mean, train_set.x2_mean, train_set.z_mean,
                                      train_set.x1_std, train_set.x2_std, train_set.z_std)
    elif args.data_set == "delay_lorenz_simple_all":
        folder, data_paths = get_delay_lorenz_path()
        train_set = DelayLorenzSimpleDataset(args, data_paths[0])
        val_set = DelayLorenzSimpleDataset(args, data_paths[1], train_set.U_z)
        test_set = DelayLorenzSimpleDataset(args, data_paths[2], train_set.U_z)
    elif args.data_set == "delay_lorenz_simple2":
        folder, data_paths = get_delay_lorenz_path()
        train_set = DelayLorenzSimple2Dataset(args, data_paths[0])
        val_set = DelayLorenzSimple2Dataset(args, data_paths[1], train_set.U_z)
        test_set = DelayLorenzSimple2Dataset(args, data_paths[2], train_set.U_z)
    elif args.data_set == "pendulum":
        folder, data_paths = get_pendulum_path()
        train_set = PendulumDataset(args, data_paths[0])
        val_set = PendulumDataset(args, data_paths[1], train_set.x_mean, train_set.x_std)
        val_set = train_set
        test_set = PendulumDataset(args, data_paths[2], train_set.x_mean, train_set.x_std)
    elif args.data_set == "arousal1":
        data_paths = get_arousal1_path()
        train_percent, _ = get_arousal1_partition()
        brain, pupil = np.load(data_paths[1]), np.load(data_paths[2])
        train_idx = int(train_percent * len(brain))
        train_set = Arousal1Dataset(brain[:train_idx], pupil[:train_idx])
        val_set = Arousal1Dataset(brain[train_idx:], pupil[train_idx:])
        test_set = None
    return train_set, val_set, test_set

def load_model(cp_path, device, net, optim=None, scheduler=None):
    checkpoint = torch.load(cp_path, map_location="cuda:" + str(device))
    net.load_state_dict(checkpoint['model'])
    net.to(device)
    if optim is not None:
        optim.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    initial_e = checkpoint['epoch']
    return net, optim, scheduler, initial_e

def make_model(args):
    if args.model == 'VAE':
        from src.models.VAE import Net
    elif args.model == 'TempVAE':
        from src.models.TempVAE import Net
    elif args.model == 'TempVAE22':
        from src.models.TempVAE22 import Net
    elif args.model == 'TempVAE33':
        from src.models.TempVAE33 import Net
    elif args.model == 'SingleTempVAE33':
        from src.models.SingleTempVAE33 import Net
    elif args.model == 'TempVAE222':
        from src.models.TempVAE222 import Net
    elif args.model == 'TempVAESimple':
        from src.models.TempVAESimple import Net
    elif args.model == 'TempVAESimple2':
        from src.models.TempVAESimple2 import Net
    elif args.model == 'TempVAESingle':
        from src.models.TempVAESingle import Net
    elif args.model == 'TempVAE2':
        from src.models.TempVAE2 import Net
    elif args.model == 'HyperTempVAE':
        from src.models.HyperTempVAE import Net
    elif args.model == 'HyperTempVAE2':
        from src.models.HyperTempVAE2 import Net
    elif args.model == 'HyperTempVAE3':
        from src.models.HyperTempVAE3 import Net
    elif args.model == 'HyperTempVAE33':
        from src.models.HyperTempVAE33 import Net
    elif args.model == 'HyperTempVAE44':
        from src.models.HyperTempVAE44 import Net
    elif args.model == 'HyperTempVAE55':
        from src.models.HyperTempVAE55 import Net
    elif args.model == 'RKTempVAE44':
        from src.models.RKTempVAE44 import Net
    elif args.model == 'RKTempVAE44Flex':
        from src.models.RKTempVAE44Flex import Net
    elif args.model == 'RKTempVAE55':
        from src.models.RKTempVAE55 import Net
    elif args.model == 'BBHSimple':
        from src.models.BBHSimple import Net
    elif args.model == 'HyperSimple':
        from src.models.HyperSimple import Net
    #elif args.model == 'BBH-ESindy-VAE':
    #    from src.models.BBH-ESindy-VAE import Net
    elif args.model == 'ESindyVAE':
        from src.models.ESindyVAE import Net
    elif args.model == 'ESindyAE':
        from src.models.ESindyAE import Net
    elif args.model == 'ESindyAESimple':
        from src.models.ESindyAESimple import Net
    elif args.model == 'HyperESindyAE':
        from src.models.HyperESindyAE import Net
    elif args.model == 'HyperAE':
        from src.models.HyperAE import Net
    elif args.model == 'SindyDerivative':
        from src.models.SindyDerivative import Net
    return Net(args)

def get_lorenz_path():
    folder = "data/lorenz/"
    return folder, get_folder_train_val_test(folder)

def get_stochastic_lorenz_path():
    folder = "data/stochastic_lorenz/"
    return folder, get_folder_train_val_test(folder)

def get_lorenz2_path():
    folder = "data/lorenz2/"
    return folder, get_folder_train_val_test(folder)

def get_lorenz3_path():
    folder = "data/lorenz3/"
    return folder, get_folder_train_val_test(folder)

def get_lorenz4_path():
    folder = "data/lorenz4/"
    return folder, get_folder_train_val_test(folder)

def get_lorenz5_path():
    folder = "data/lorenz5/"
    return folder, get_folder_train_val_test(folder)

def get_delay_lorenz_path():
    folder = "data/delay_lorenz/"
    return folder, get_folder_train_val_test(folder)

def get_delay_lorenz_noise_path():
    folder = "data/delay_lorenz_noise/"
    return folder, get_folder_train_val_test(folder)

def get_delay_lorenz_big_path():
    folder = "data/delay_lorenz_big/"
    return folder, get_folder_train_val_test(folder)

def get_pendulum_path():
    folder = "data/pendulum/" 
    return folder, get_folder_train_val_test(folder)

def get_arousal1_path():
    folder = "data/arousal1/" 
    return folder, (folder + "brain1_embedded.npy", folder + "pupil1_embedded.npy")

def get_arousal1_partition():
    train_percent = 0.75
    val_percent = 0.25
    return train_percent, val_percent

def get_general_path(args):
    path = args.data_set + "/" + args.model + "/" + args.date + "/"
    if args.session_name is None:
        path += "scale-" + str(args.scale_data)
        path += "-jm-" + str(args.just_mean)
        path += "-beta-" + str(args.beta)
        path += "-lr-" + str(args.learning_rate)
        path += "-ed-" + str(args.embedding_dim)
        path += '-hd-' + str(args.hidden_dim)
        path += '-tau-' + str(args.tau)
        path += '-htl-' + str(args.hankel_trajectory_length)
        if args.session_name2 is not None:
            path += '-' + args.session_name2
    else:
        path += args.session_name
    path += "/"
    return path

def get_folder_train_val_test(folder):
    return (folder + "train.npy", folder + "val.npy", folder + "test.npy")

def get_checkpoint_path(args):
    cp_folder = args.model_folder + get_general_path(args)
    return cp_folder + 'checkpoint.pt', cp_folder

def get_args_path(args):
    args_folder = args.model_folder +get_general_path(args)
    return args_folder + "args.txt", args_folder

def get_tb_path(args):
    train_name = args.tensorboard_folder + get_general_path(args) + "train"
    test_name = args.tensorboard_folder + get_general_path(args) + "val"
    return train_name, test_name

def get_experiments_path(args):
    return args.experiments + get_general_path(args)

def make_folder(folder):
    if not os.path.isdir(folder):
        os.system("mkdir -p " + folder)

def print_folders(do_print, cp_folder=None, exp_folder=None, tb_folder=None):
    if do_print == 1:
        if cp_folder is not None:
            print("Checkpoints saved at:        ", cp_folder)
        if exp_folder is not None:
            print("Experiment results saved at: ", exp_folder)
        if tb_folder is not None:
            print("Tensorboard logs saved at:   ", tb_folder)
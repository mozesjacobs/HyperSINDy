import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from src.utils.other import save_model
from src.utils.exp_utils import sample_trajectory, sample_ensemble_trajectory, get_equations
from src.utils.plotting import plot_trajectory

def train(net, args, hyperparams, optim, scheduler, trainloader, trainset, 
          board, cp_path, initial_epoch, device):
    """Trains the network.

    Trains the given network using the given dataloader and parameters. Logs 
    results and saves the model over the course of training.

    Args:
        net: The network to be trained (subclass of nn.Module()).
        args: The argparser object return by parse_args() in the file
            cmd_line.py.
        hyperparams: The argparser object returned by parse_hyperparams() in 
            the file cmd_line.py
        optim: The torch.optim optimizer used to train net.
        scheduler: The torch.optim.lr_scheduler used to tune the learning rate 
            of optim.
        trainloader: The torch dataloader that net will be trained with.
        trainset: The torch dataset that trainloader was created with. The
            network will be evaluated throughout training using trainset.
        board: The tensorboard SummaryWriter used to log results throughout
            the course of training.
        cp_path: The str path to save net, optim, scheduler, and epoch 
            throughout the course of training.
        initial_epoch: The int epoch to start training at (only affects logging
            of results).
        device: The cpu or gpu device to train the network on. To train on cpu,
            device must be "cpu". To train on gpu, specify which gpu to use as
            an integer (i.e.: 0 or 1 or 2 or 3). 

    Returns:
        None
    """
    
    beta = args.beta_init
    beta_max = hyperparams.beta
    beta_inc = args.beta_inc
    if beta_inc is None:
        beta_inc = beta_max / 100.0

    for epoch in range(initial_epoch, hyperparams.epochs + initial_epoch):
        # one train step
        recons, regs = train_epoch(
            net, args.model, trainloader, optim, beta,
            hyperparams.weight_decay, device, hyperparams.clip)

        # log losses
        log_losses(args.model, board, recons / len(trainloader), regs / len(trainloader), epoch)

        # threshold
        update_threshold_mask(net, args.model, hyperparams.hard_threshold,
                              hyperparams.threshold_interval, epoch, device,
                              beta, beta_max)

        # save
        if (epoch % args.checkpoint_interval == 0) and (epoch != 0):
            save_model(cp_path, net, optim, scheduler, epoch)

        # save
        if (epoch % args.eval_interval == 0) and (epoch != 0):
            eval_model(net.eval(), args, board, trainset, device, epoch)

        scheduler.step()
        beta = update_beta(beta, beta_inc, beta_max)

    save_model(cp_path, net, optim, scheduler, epoch)
    eval_model(net.eval(), args, board, trainset, device, epoch)

    #return net, optim, scheduler

def train_epoch(net, model_type, trainloader, optim, beta, weight_decay,
                device, clip):
    """Trains the network for one epoch.

    Trains the network for one pass over the given dataloader using the given
    parameters.

    Args:
        net: The network to be trained (subclass of nn.Module()).
        model_type: The str name of the model ({"HyperSINDy", "ESINDy",
            "SINDy"}). Equivalent to the model_type arguments in parse_args
            from cmd_line.py
        trainloader: The torch dataloader that net will be trained with.
        hyperparams: The argparser object returned by parse_hyperparams() in 
            the file cmd_line.py
        optim: The torch.optim optimizer used to train net.
        beta: A float value denoting the strength of the KL divergence term in
            the HyperSINDy loss function.
        weight_decay: A float value denoting the strength of the regularization
            term in the SINDy or ESINDy loss function.
        device: The cpu or gpu device to train the network on. To train on cpu,
            device must be "cpu". To train on gpu, specify which gpu to use as
            an integer (i.e.: 0 or 1 or 2 or 3). 
        clip: The float value to clip the gradients to during training (use
            None to disable gradient clipping).
        
    Returns:
        A tuple (float_a, float_b) where float_a is the sum of the derivative
        calculation loss over all the batches in the dataloader and float_b
        is the sum of the regularization term over all the batches in the
        dataloader.
    """

    # train mode
    net = net.train()
    
    recons, regs = 0, 0
    for i, (x, x_lib, x_dot) in enumerate(trainloader):
        x_dot = x_dot.type(torch.FloatTensor).to(device)

        # one gradient step
        if model_type == "HyperSINDy":
            recon, reg = train_hyper(net, optim, x, x_lib, x_dot, beta,
                                     device, clip)
        elif model_type == "ESINDy":
            recon, reg = train_ensemble(net, optim, x, x_lib, x_dot, weight_decay,
                                        device, clip)
        elif model_type == "SINDy":
            recon, reg = train_sindy(net, optim, x, x_lib, x_dot, weight_decay,
                                     device, clip)

        recons += recon
        regs += reg
    return recons, regs

def train_hyper(net, optim, x, x_lib, x_dot, beta, device, clip):
    """Trains the hypernetwork on one batch of data.

    Feeds the hypernetwork the given data batch, calculates the loss, and
    performs one gradient step.

    Args:
        net: The network to be trained (subclass of nn.Module()).
        optim: The torch.optim optimizer used to update the parameters of net.
        x: The raw data (torch.Tensor) to evaluate the model on. Should have
            shape (batch_size x z_dim).
        x_lib: The result (torch.Tensor) of calling sindy_library from
            model_utils.py on x. Should have shape (batch_size x library_dim).
        x_dot: The corresponding derivatives (torch.Tensor) of x. Should have
            shape (batch_size x z_dim).
        beta: A float value denoting the strength of the KL divergence term in
            the HyperSINDy loss function.
        device: The cpu or gpu device to train the network on. To train on cpu,
            device must be "cpu". To train on gpu, specify which gpu to use as
            an integer (i.e.: 0 or 1 or 2 or 3).
        clip: The float value to clip the gradients to during training (use
            None to disable gradient clipping).
        
    Returns:
        A tuple (float_a, float_b) where float_a is the the derivative
        calculation loss over the given data batch and float_b is the KL
        divergence term in the loss function.
    """"
    x_dot_pred, sindy_coeffs = net(x, x_lib, device)
    recon = ((x_dot_pred - x_dot) ** 2).sum(1).mean()
    kld = net.kl(sindy_coeffs)
    loss = recon + kld * beta
    optim.zero_grad()
    loss.backward()
    if clip is not None:
        nn.utils.clip_grad_norm_(net.parameters(), clip)
    optim.step()
    return recon.item(), kld.item()

def train_ensemble(net, optim, x, x_lib, x_dot, weight_decay, device, clip):
    """Trains the ensemble of SINDy models on one batch of data.

    Feeds the ensemble the given data batch, calculates the loss, and
    performs one gradient step.

    Args:
        net: The network to be trained (subclass of nn.Module()).
        optim: The torch.optim optimizer used to update the parameters of net.
        x: The raw data (torch.Tensor) to evaluate the model on. Should have
            shape (batch_size x num_ensemble x z_dim).
        x_lib: The result (torch.Tensor) of calling sindy_library from
            model_utils.py on x. Should have shape
            (batch_size x num_ensemble x library_dim).
        x_dot: The corresponding derivatives (torch.Tensor) of x. Should have
            shape (batch_size x num_ensemble x z_dim).
        weight_decay: A float value denoting the strength of the regularization
            term in the ensemble loss function.
        device: The cpu or gpu device to train the network on. To train on cpu,
            device must be "cpu". To train on gpu, specify which gpu to use as
            an integer (i.e.: 0 or 1 or 2 or 3).
        clip: The float value to clip the gradients to during training (use
            None to disable gradient clipping).
        
    Returns:
        A tuple (float_a, float_b) where float_a is the the derivative
        calculation loss over the given data batch and float_b is the
        regularization term in the loss function.
    """"
    x_dot_pred, sindy_coeffs = net(x, x_lib, device)
    recon = ((x_dot_pred - x_dot) ** 2).sum(2).mean()
    if net.prior == 'normal':
        reg = (sindy_coeffs ** 2).sum((1, 2)).mean() * weight_decay
    elif net.prior == 'laplace':
        reg = torch.abs(sindy_coeffs).sum((1, 2)).mean() * weight_decay
    else:
        print("ERROR: net.prior must be normal or laplace, not " + str(net.prior))
        exit()
    loss = recon + reg
    optim.zero_grad()
    loss.backward()
    if clip is not None:
        nn.utils.clip_grad_norm_(net.parameters(), clip)
    optim.step()
    return recon.item(), reg.item()

def train_sindy(net, optim, x, x_lib, x_dot, weight_decay, device, clip):
    """Trains the SINDy model on one batch of data.

    Feeds the SINDy model the given data batch, calculates the loss, and
    performs one gradient step.

    Args:
        net: The network to be trained (subclass of nn.Module()).
        optim: The torch.optim optimizer used to update the parameters of net.
        x: The raw data (torch.Tensor) to evaluate the model on. Should have
            shape (batch_size x z_dim).
        x_lib: The result (torch.Tensor) of calling sindy_library from
            model_utils.py on x. Should have shape (batch_size x library_dim).
        x_dot: The corresponding derivatives (torch.Tensor) of x. Should have
            shape (batch_size x z_dim).
        weight_decay: A float value denoting the strength of the regularization
            term in the SINDy loss function.
        device: The cpu or gpu device to train the network on. To train on cpu,
            device must be "cpu". To train on gpu, specify which gpu to use as
            an integer (i.e.: 0 or 1 or 2 or 3).
        clip: The float value to clip the gradients to during training (use
            None to disable gradient clipping).
        
    Returns:
        A tuple (float_a, float_b) where float_a is the the derivative
        calculation loss over the given data batch and float_b is the
        regularization term in the loss function.
    """"
    x_dot_pred, sindy_coeffs = net(x, x_lib, device)
    recon = ((x_dot_pred - x_dot) ** 2).sum(1).mean()
    if net.prior == 'normal':
        reg = (sindy_coeffs ** 2).sum((0, 1)) * weight_decay
    elif net.prior == 'laplace':
        reg = torch.abs(sindy_coeffs).sum((0, 1)) * weight_decay
    else:
        print("ERROR: net.prior must be normal or laplace, not " + str(net.prior))
        exit()
    loss = recon + reg
    optim.zero_grad()
    loss.backward()
    if clip is not None:
        nn.utils.clip_grad_norm_(net.parameters(), clip)
    optim.step()
    return recon.item(), reg.item()

def update_threshold_mask(net, model_type, threshold, threshold_timer, epoch, device, beta, beta_max):
    """Updates the threshold mask based on coefficient values.

    If coefficients are less than the given threshold, sets the corresponding
    value in the given networks threshold mask to 0. For HyperSINDy, samples
    a batch of coefficients and uses the mean over the batch as the
    the coefficients to be judged.

    Args:
        net: The network (subclass of nn.Module()) to be thresholded
        model_type: The str name of the model ({"HyperSINDy", "ESINDy",
            "SINDy"}). Equivalent to the model_type arguments in parse_args
            from cmd_line.py
        threshold: The minimum float value for coefficients to be included.
            If a coefficient is less then this threshold, it will be
            permanently zero-d out.
        threshold_timer: The epoch interval (int) to do thresholding.
        epoch: The current epoch (int) during training. If 
            epoch % threshold_timer != 0, will not threshold.
        device: The cpu or gpu device to train the network on. To train on cpu,
            device must be "cpu". To train on gpu, specify which gpu to use as
            an integer (i.e.: 0 or 1 or 2 or 3). 
        beta: A float value denoting the strength of the KL divergence term in
            the HyperSINDy loss function.
        beta_max: The maximum beta value to be reached during training. For
            HyperSINDy, if beta =/= beta_max, will not perform thresholding.
            Does not affect ESINDy or SINDy.

    Returns:
        None
    """
    with torch.no_grad():
        if (epoch % threshold_timer == 0) and (epoch != 0):
            if (model_type == "HyperSINDy"):
                if (beta == beta_max):
                    net.update_threshold_mask(threshold, device)
            else:
                net.update_threshold_mask(threshold)

def log_losses(model_type, board, recon, reg, epoch):
    """Logs the losses in tensorboard.

    Updates the given reconstruction and regularization loss in the given
    SummaryWriter.

    Args:
        model_type: The str name of the model ({"HyperSINDy", "ESINDy",
            "SINDy"}). Equivalent to the model_type arguments in parse_args
            from cmd_line.py
        board: The tensorboard SummaryWriter used to log results throughout
            the course of training.
        recon: A float representing the error between the predicted and ground
            truth derivatives.
        reg: A float representing the KL divergence or the L1/L2 loss of
            the coefficients.
        epoch: The current epoch (int) during training. If 
            epoch % threshold_timer != 0, will not threshold.
            
    Returns:
        None
    """
    board.add_scalar("Loss/(x_dot_pred - x_dot) ** 2", recon, epoch)
    if model_type == "HyperSINDy":
        board.add_scalar("Loss/kld", reg, epoch)
    elif model_type == "ESINDy" or model_type == "SINDy":
        board.add_scalar("Loss/reg", reg, epoch)

def update_beta(beta, beta_increment, beta_max):
    """Updates the beta value.

    Increases beta by the given increment. If incrementing beta would make it
    exceed the given max, sets beta equal to the max.

    Args:
        beta: A float value denoting the strength of the KL divergence term in
            the HyperSINDy loss function.
        beta: A float denoting the amount to increment beta by.
        beta_max: The maximum beta value to be reached during training.
            
    Returns:
        A float of the updated beta value.
    """
    beta += beta_increment
    if beta > beta_max:
        beta = beta_max
    return beta

def eval_model(net, args, board, trainset, device, epoch):
    """Evaluates the network.

    Generates sample trajectories and logs them in tensorboard. Logs
    the current discovered equations in tensorboard.

    Args:
        net: The network (subclass of nn.Module()) to be evaluated.
        args: The argparser object return by parse_args() in the file
            cmd_line.py.
        board: The tensorboard SummaryWriter to log results with.
        trainset: The torch dataset to evaluate the model on.
        device: The cpu or gpu device to train the network on. To train on cpu,
            device must be "cpu". To train on gpu, specify which gpu to use as
            an integer (i.e.: 0 or 1 or 2 or 3). 
        epoch: An int representing the current epoch during training.
            
    Returns:
        None
    """
    if args.model == "HyperSINDy" or args.model == "SINDy":
        # sample trajectory
        z = sample_trajectory(net, device, trainset.x[0].numpy(),
                              args.exp_batch_size, args.dt, args.exp_timesteps)
        # plot trajectory
        plot_trajectory(board, epoch, trainset.x.numpy(), z)
    elif args.model == "ESINDy":
        # sample trajectory
        z = sample_ensemble_trajectory(net, device, trainset.x_true[0].numpy(),
                            args.exp_batch_size, args.dt, args.exp_timesteps)
        # plot trajectory
        plot_trajectory(board, epoch, trainset.x_true.numpy(), z)
    else:
        print("ERROR: args.model must be HyperSINDy, SINDy, or ESINDy, not " + args.model + ".")

    
    # get equations
    equations = get_equations(net, args.model, device,
                              args.z_dim, args.poly_order,
                              args.include_constant, args.include_sine)

    if args.model == "HyperSINDy" or args.model == "ESINDy":
        eq_mean = str(equations[1]) + "  \n" + str(equations[2]) + "  \n" + str(equations[3])
        eq_std = str(equations[5]) + "  \n" + str(equations[6]) + "  \n" + str(equations[7])
        if args.z_dim == 5:
            eq_mean = str(equations[1]) + "  \n" + str(equations[2]) + "  \n" + str(equations[3]) + "  \n" + str(equations[4]) + "  \n" + str(equations[5])
            eq_std = str(equations[7]) + "  \n" + str(equations[8]) + "  \n" + str(equations[9]) + "  \n" + str(equations[10]) + "  \n" + str(equations[11])
        board.add_text(tag="Equations/mean", text_string=eq_mean, global_step=epoch, walltime=None)
        board.add_text(tag="Equations/std", text_string=eq_std, global_step=epoch, walltime=None)
    elif args.model == "SINDy":
        eq_mean = str(equations[1]) + "  \n" + str(equations[2]) + "  \n" + str(equations[3])
        if args.z_dim == 5:
            eq_mean = str(equations[1]) + "  \n" + str(equations[2]) + "  \n" + str(equations[3]) + "  \n" + str(equations[4]) + "  \n" + str(equations[5])
        board.add_text(tag="Equations/SINDy", text_string=eq_mean, global_step=epoch, walltime=None)
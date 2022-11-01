import argparse

def hyperparameter_grid():
    settings = {}
    settings['num_samples'] = 6
    settings['beta'] = [0.1, 1.0, 10.0]
    settings['sindy_weight_decay'] = [0.01, 0.1]
    #settings['sindy_weight_decay'] = lambda _ : 10 ** np.random.randint(-4, -1 + 1)
    settings['noise_coef_threshold'] = [0.1]
    settings['sindy_coef_threshold'] = [0.1]
    return settings


def parse_args():
    
    parser = argparse.ArgumentParser(description="Template")

    # base folders
    parser.add_argument('-EX', '--experiments', default='./experiments/', type=str, help="Output folder for experiments")
    parser.add_argument('-MF', '--model_folder', default='./trained_models/', type=str, help="Output folder for experiments")
    parser.add_argument('-TB', '--tensorboard_folder', default='./tb_runs/', type=str, help="Output folder for tensorboard")

    # saving specifics
    parser.add_argument('-sess', '--session_name', default='debug', type=str, help="Appended second to last part of file names")
    parser.add_argument('-DAT', '--date', default="10-30-22", type=str, help="The date"),
    parser.add_argument('-M',  '--model', default="HyperSINDy3", type=str, help="Model to use")
    parser.add_argument('-DT', '--data_set', default="lorenz3", type=str, help="Which dataset to use (lorenz)")
    parser.add_argument('-NOISET', '--noise_type', default='statesinstate', type=str, help='Type of state-dependent noise (x, sinz)')
    parser.add_argument('-NOISES', '--noise_scale', default=2.5, type=float, help='Scale of noise in data. Review data folder.')

    # network parameters
    parser.add_argument('-HD', '--hidden_dim', default=64, type=str, help="Dimension of hidden layers in autoencoder")
    parser.add_argument('-NOD',  '--noise_dim', default=25, type=int, help="Noise vector dimension")
    parser.add_argument('-NORM', '--norm', default='none', type=str, help="Norm used in hypernet (batch, layer). If neither of those, no norm used.")
    parser.add_argument('-SBS', '--statistic_batch_size', default=500, type=str, help="Batch size to sample for statistics")

    # sindy parameters
    parser.add_argument('-Z', '--z_dim', default=3, type=int, help="Size of latent vector")
    parser.add_argument('-PO', '--poly_order', default=3, type=int, help="Size of theta library for SINDy")
    parser.add_argument('-INCS', '--include_sine', default=False, type=bool, help="IFF True, includes sine in SINDy library")
    parser.add_argument('-INCMS', '--include_mult_sine', default=True, type=bool, help="IFF True, includes state * sine(state) in SINDy library")
    parser.add_argument('-INCC', '--include_constant', default=False, type=bool, help="IFF True, includes sine in SINDy library")

    # training parameters
    parser.add_argument('-E', '--epochs', default=501, type=float, help="Number of epochs to train for")
    parser.add_argument('-LR', '--learning_rate', default=1e-2, type=float, help="Learning rate")
    parser.add_argument('-ARE', '--adam_regularization', default=0, type=float, help="Regularization to use in ADAM optimizer")
    parser.add_argument('-GF', '--gamma_factor', default=0.999, type=float, help="Learning rate decay factor")
    parser.add_argument('-BS', '--batch_size', default=250, type=float, help="Batch size")
    parser.add_argument('-C', '--clip', default=None, type=float, help="Gradient clipping value during training (None for no clipping)")
    parser.add_argument('-THRESHI', '--threshold_interval', default=100, type=float, help="Epoch interval to threshold sindy and noise during training")
    parser.add_argument('-CPI', '--checkpoint_interval', default=25, type=float, help="Epoch interval to save model during training")
    parser.add_argument('-BINIT', '--beta_init', default=0.01, type=float, help="Inital beta value")
    parser.add_argument('-BINCR', '--beta_inc', default=None, type=float, help="Beta increment per epoch till beta max. If none, = beta_max / 100")

    # dataset parameters
    parser.add_argument('-TDT', '--delta_t', default=0.01, type=float, help='Time change in dataset')
    parser.add_argument('-ND', '--norm_data', default=False, type=bool, help='Iff true, normalized data to N(0, 1)')
    parser.add_argument('-SD', '--scale_data', default=0.0, type=int, help='Scales the data values (after normalizing).')

    # other
    parser.add_argument('-D', '--device', default=3, type=int, help='Which GPU to use')
    parser.add_argument('-LCP', '--load_cp', default=0, type=int, help='If 1, loads the model from the checkpoint. If 0, does not')
    parser.add_argument('-PF', '--print_folder', default=1, type=int, help='Iff true, prints the folder for different logs')

    # experiment parameters
    parser.add_argument('-EBS', '--exp_batch_size', default=5, type=int, help='Batch size for experiment')
    parser.add_argument('-ETS', '--exp_timesteps', default=5000, type=int, help='Number of timesteps per trajectory')
    parser.add_argument('-EDT', '--exp_dt', default=0.01, type=int, help='dt for experiment')
    
    return parser.parse_args()
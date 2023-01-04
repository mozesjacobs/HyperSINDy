import argparse


# These are hyperparameters that get logged into tensorboard
def parse_hyperparams():
    parser = argparse.ArgumentParser(description="Template")
    parser.add_argument('-B', '--beta', default=1.0, type=float, help="KL divergence weight in loss (only for HyperSINDy)")
    parser.add_argument('-WD', '--weight_decay', default=1e-3, type=float, help="Weight decay for sindy coefficients (only for SINDy)")
    parser.add_argument('-ST', '--soft_threshold', default=0.05, type=float, help="Soft threshold to 0 coefficient samples. 0 to disable.")
    parser.add_argument('-HT', '--hard_threshold', default=0.1, type=float, help="Hard threshold to permanently 0 coefficients out. Updated every threshold_interval epochs. 0 to disable.")
    parser.add_argument('-TI', '--threshold_interval', default=100, type=float, help="Epoch interval in training to permanently threshold sindy coefs")
    parser.add_argument('-ND', '--noise_dim', default=25, type=int, help="Noise vector dimension for HyperSINDy")
    parser.add_argument('-HD', '--hidden_dim', default=128, type=str, help="Dimension of hidden layers hypernet")
    parser.add_argument('-E', '--epochs', default=751, type=float, help="Number of epochs to train for")
    parser.add_argument('-LR', '--learning_rate', default=1e-3, type=float, help="Learning rate")
    parser.add_argument('-BS', '--batch_size', default=250, type=float, help="Batch size")
    parser.add_argument('-AR', '--adam_reg', default=0, type=float, help="Regularization to use in ADAM optimizer")
    parser.add_argument('-C', '--clip', default=1.0, type=float, help="Gradient clipping value during training (None for no clipping)")
    parser.add_argument('-P', '--prior', default="normal", type=str, help="Prior to regularize to. Options: laplace, normal")
    return parser.parse_args()


def parse_args():
    
    parser = argparse.ArgumentParser(description="Template")

    # base folders
    parser.add_argument('-DF', '--data_folder', default='./data/', type=str, help="Base folder where all data is stored")
    parser.add_argument('-EX', '--experiments', default='./experiments/', type=str, help="Output folder for experiments")
    parser.add_argument('-MF', '--model_folder', default='./trained_models/', type=str, help="Output folder for experiments")
    parser.add_argument('-TB', '--tensorboard_folder', default='./runs/5d_lorenz/', type=str, help="Output folder for tensorboard")

    # saving specifics
    parser.add_argument('-sess', '--session_name', default='1', type=str, help="Appended to last part of file names")
    parser.add_argument('-DAT', '--date', default="1-04-23", type=str, help="The date"),
    parser.add_argument('-M',  '--model', default="HyperSINDy", type=str, help="Model to use")
    parser.add_argument('-DS', '--dataset', default="lorenz", type=str, help="Which dataset to use (lorenz)")
    parser.add_argument('-NOISET', '--noise_type', default='x', type=str, help='Type of state-dependent noise (x, sinz)')
    parser.add_argument('-NOISES', '--noise_scale', default=1.0, type=float, help='Scale of noise in data. Review data folder.')

    # sindy parameters
    parser.add_argument('-Z', '--z_dim', default=5, type=int, help="Size of latent vector")
    parser.add_argument('-PO', '--poly_order', default=3, type=int, help="Size of theta library for SINDy")
    parser.add_argument('-INCC', '--include_constant', default=False, type=bool, help="IFF True, includes sine in SINDy library")
    parser.add_argument('-INCS', '--include_sine', default=False, type=bool, help="IFF True, includes sine in SINDy library")
    
    # training parameters
    parser.add_argument('-GF', '--gamma_factor', default=0.999, type=float, help="Learning rate decay factor")
    parser.add_argument('-CPI', '--checkpoint_interval', default=25, type=float, help="Epoch interval to save model during training")
    parser.add_argument('-EI', '--eval_interval', default=50, type=float, help="Epoch interval to evalate model during training")
    parser.add_argument('-BINIT', '--beta_init', default=0.01, type=float, help="Inital beta value")
    parser.add_argument('-BINCR', '--beta_inc', default=None, type=float, help="Beta increment per epoch till beta max. If none, = beta_max / 100")

    # dataset parameters
    parser.add_argument('-ND', '--norm_data', default=False, type=bool, help='Iff true, normalizes data to N(0, 1)')
    parser.add_argument('-SD', '--scale_data', default=0.0, type=int, help='Scales the data values (after normalizing).')

    # experiment parameters
    parser.add_argument('-EBS', '--exp_batch_size', default=5, type=int, help='Batch size for experiments')
    parser.add_argument('-ETS', '--exp_timesteps', default=10000, type=int, help='Number of timesteps per trajectory')

    # other
    parser.add_argument('-D', '--device', default=0, type=int, help='Which GPU to use')
    parser.add_argument('-LCP', '--load_cp', default=0, type=int, help='If 1, loads the model from the checkpoint. If 0, does not')
    parser.add_argument('-PF', '--print_folder', default=1, type=int, help='Iff true, prints the folder for different logs')
    parser.add_argument('-DT', '--dt', default=0.0025, type=float, help='Time change in dataset')
    parser.add_argument('-SBS', '--statistic_batch_size', default=500, type=str, help="Default batch size to sample")
    
    return parser.parse_args()
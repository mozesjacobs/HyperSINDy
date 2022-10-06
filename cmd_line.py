import argparse


def parse_args():
    
    parser = argparse.ArgumentParser(description="Template")

    # anything that affects the name of the saved folders (for checkpoints, experiments, tensorboard)
    # path saved as: initial_folder/data_set/model/session_name
    # where initial_folder is experiments, model_folder, or tensorboard_folder
    parser.add_argument('-sess', '--session_name', default=None, type=str, help="If not None, uses this.")
    parser.add_argument('-sess2', '--session_name2', default=None, type=str, help="Uses hyperparameters in saved name and session_name2 is appended") #10tau_20hankel, 10tau_40hankel
    parser.add_argument('-DAT', '--date', default="9-15-22", type=str, help="The date"), # 8-29-22, 8-29-22
    parser.add_argument('-M',  '--model', default="RKTempVAE44Flex", type=str, help="Model to use") # TempVAE33, TempVAE33
    parser.add_argument('-EX', '--experiments', default='./experiments/', type=str, help="Output folder for experiments")
    parser.add_argument('-MF', '--model_folder', default='./trained_models/', type=str, help="Output folder for experiments")
    parser.add_argument('-TB', '--tensorboard_folder', default='./tb_runs/', type=str, help="Output folder for tensorboard")
    parser.add_argument('-DT', '--data_set', default='stochastic_lorenz_flex', type=str, help="Which dataset to use (lorenz)"), # delay_lorenz, delay_lorenz_big

    # network parameters
    parser.add_argument('-Z', '--z_dim', default=3, type=int, help="Size of latent vector")
    parser.add_argument('-ED', '--embedding_dim', default=30, type=int, help="Size of latent vector")
    parser.add_argument('-U',  '--u_dim', default=10, type=int, help="Sise of u vector in Lorenz data")
    parser.add_argument('-HD', '--hidden_dim', default=128, type=str, help="Dimension of hidden layers in autoencoder")
    parser.add_argument('-MHD', '--mlp_hidden_dim', default=12, type=int, help="Hidden dimension in predicted MLP")
    parser.add_argument('-ID1',  '--in_dim_1', default=20, type=int, help="Size of u vector in Lorenz data") # 30, 30
    parser.add_argument('-ID2',  '--in_dim_2', default=20, type=int, help="Size of u vector in Lorenz data") # 30, 30
    parser.add_argument('-NOD',  '--noise_dim', default=10, type=int, help="Noise vector dimension")
    parser.add_argument('-JM', '--just_mean', default=False, type=bool, help="Iff true, uses only the mean of the distribution")  # False, False
    parser.add_argument('-B', '--beta', default=1.0, type=float, help="Weight of KL term in loss function") # 1.0, 1.0
    parser.add_argument('-PO', '--poly_order', default=3, type=int, help="Size of theta library for SINDy")
    parser.add_argument('-INCS', '--include_sine', default=False, type=bool, help="IFF True, includes sine in SINDy library")
    parser.add_argument('-INCC', '--include_constant', default=True, type=bool, help="IFF True, includes sine in SINDy library")
    parser.add_argument('-COEFR', '--coef_reg', default=0.01, type=float, help="SINDy coefficient regularization")
    parser.add_argument('-SEQT', '--sequential_threshold', default=None, type=bool, help="SINDy sequential threshold term")
    parser.add_argument('-BN', '--batch_norm', default=False, type=bool, help="IFF true, uses batch norm")
    parser.add_argument('-SMT', '--smoothness', default=0, type=float, help="Smoothness constraint weight")

    # training parameters
    parser.add_argument('-E', '--epochs', default=100, type=float, help="Number of epochs to train for")
    parser.add_argument('-LR', '--learning_rate', default=1e-3, type=float, help="Learning rate")
    parser.add_argument('-ARE', '--adam_regularization', default=1e-4, type=float, help="Regularization to use in ADAM optimizer") # 0, 0
    parser.add_argument('-GF', '--gamma_factor', default=0.999, type=float, help="Learning rate decay factor")
    parser.add_argument('-BS', '--batch_size', default=200, type=float, help="Batch size")
    parser.add_argument('-C', '--clip', default=None, type=float, help="Gradient clipping value during training (None for no clipping)")
    parser.add_argument('-TI', '--test_interval', default=1, type=float, help="Epoch interval to evaluate on val (test) data during training")
    parser.add_argument('-CPI', '--checkpoint_interval', default=1, type=float, help="Epoch interval to save model during training")

    # lorenz dataset parameters
    parser.add_argument('-TS', '--timesteps', default=1000, type=int, help='Number of timesteps per trajectory') # 1000, 5000
    parser.add_argument('-TDT', '--delta_t', default=0.02, type=float, help='Time change in dataset') # 0.02, 0.01
    parser.add_argument('-TIC', '--train_initial_conds', default=2048, type=int, help='Number of initial conditions in the training set')
    parser.add_argument('-VIC', '--val_initial_conds', default=20, type=int, help='Number of initial conditions in the validation set')
    parser.add_argument('-TEIC', '--test_initial_conds', default=100, type=int, help='Number of initial conditions in the test set')
    parser.add_argument('-NSTR', '--noise_strength', default=0, type=float, help='Strength of noise in lorenz datasaet. 0 for no noise')
    parser.add_argument('-SR', '--subsample_rate', default=1, type=int, help='Number of timestep in dataset (select 1 timestep every subsample_rate timesteps)')
    parser.add_argument('-TAU', '--tau', default=10, type=int, help='Tau for future prediction') # 10, 10
    parser.add_argument('-K', '--hankel_trajectory_length', default=20, type=int, help='Length of a snapshot trajectory in hankel matrix') # 20, 40
    parser.add_argument('-NC', '--num_comps', default=3, type=int, help='Number of components to use in svd')
    parser.add_argument('-DYN', '--dynamic', default=True, type=bool, help='If true, model decoder predicts tau into the future. If false, model reconstructs current x.')
    parser.add_argument('-ND', '--norm_data', default=True, type=bool, help='Iff true, normalized data to N(0, 1)') # True, True
    parser.add_argument('-SD', '--scale_data', default=1.0, type=int, help='Scales the data values (after normalizing).') # 1.0, 1.0
    parser.add_argument('-DIDX', '--data_idx', default=0, type=int, help='Which data trajectory to use for hankel')

    # other
    parser.add_argument('-LCP', '--load_cp', default=0, type=int, help='If 1, loads the model from the checkpoint. If 0, does not')
    parser.add_argument('-D', '--device', default=1, type=int, help='Which GPU to use')
    parser.add_argument('-PF', '--print_folder', default=1, type=int, help='Iff true, prints the folder for different logs')

    # experiment parameters
    parser.add_argument('-NP', '--num_plot', default=5, type=int, help='Number of trajectories to plot in the gif')
    parser.add_argument('-GD', '--gif_duration', default=0.25, type=int, help='Duration of each frame in the gif')
    parser.add_argument('-EJM', '--exp_just_mean', default=False, type=bool, help="Iff true, uses only the mean of the distribution")
    parser.add_argument('-ETS', '--exp_timesteps', default=100, type=int, help='Number of timesteps per trajectory')
    parser.add_argument('-ZPS', '--z_plot_start', default=0, type=int, help='Number of timesteps per trajectory')
    parser.add_argument('-ZPE', '--z_plot_end', default=None, type=int, help='Number of timesteps per trajectory')
    parser.add_argument('-CPN', '--custom_plot_name', default='', type=str, help='Appends to end of plot name after exp# and before .extension')
    parser.add_argument('-V1', '--v1', default=None, type=float, help='View angle 1 for 3D plot')
    parser.add_argument('-V2', '--v2', default=None, type=float, help='View angle 2 for 3D plot')
    parser.add_argument('-ERPO', '--exp_return_post', default=False, type=bool, help="If true, returns post sample. If false, prior. Just for exp9.")

    # which experiments to run
    parser.add_argument('-RA', '--run_all', default=0, type=int, help='IFF 1, runs all the experiments.')
    parser.add_argument('-E1', '--e1', default=0, type=int, help='IFF 1, runs experiment 1')
    parser.add_argument('-E2', '--e2', default=0, type=int, help='IFF 1, runs experiment 2')
    parser.add_argument('-E3', '--e3', default=0, type=int, help='IFF 1, runs experiment 3')
    parser.add_argument('-E4', '--e4', default=0, type=int, help='IFF 1, runs experiment 4')
    parser.add_argument('-E5', '--e5', default=0, type=int, help='IFF 1, runs experiment 5')
    parser.add_argument('-E6', '--e6', default=0, type=int, help='IFF 1, runs experiment 6')
    parser.add_argument('-E7', '--e7', default=0, type=int, help='IFF 1, runs experiment 7')
    parser.add_argument('-E8', '--e8', default=0, type=int, help='IFF 1, runs experiment 8')
    parser.add_argument('-E9', '--e9', default=0, type=int, help='IFF 1, runs experiment 9')
    parser.add_argument('-E10', '--e10', default=0, type=int, help='IFF 1, runs experiment 10')
    parser.add_argument('-E11', '--e11', default=0, type=int, help='IFF 1, runs experiment 11')
    parser.add_argument('-E12', '--e12', default=0, type=int, help='IFF 1, runs experiment 12')
    parser.add_argument('-E13', '--e13', default=0, type=int, help='IFF 1, runs experiment 13')
    parser.add_argument('-E14', '--e14', default=0, type=int, help='IFF 1, runs experiment 13')
    
    return parser.parse_args()
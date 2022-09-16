import argparse
from train import TrainVAE

# Arguments
parser = argparse.ArgumentParser(description='Train VAE')

parser.add_argument('--exp_detail', default='Train VAE', type=str)
parser.add_argument('--gpu_num', default=0, type=int)
parser.add_argument('--seed', default=100, type=int)

# Training parameters
parser.add_argument('--n_epochs', default=200, type=int)
parser.add_argument('--step_size', default=50, type=int)
parser.add_argument('--gamma', default=0.5, type=float)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--kld_weight', default=0.01, type=float)

parser.add_argument('--n_downsample', default=2, type=int)
parser.add_argument('--n_upsample', default=2, type=int)
parser.add_argument('--multiple', default=2, type=int)  # 2~

parser.add_argument('--dataset_name', default='SEM1', type=str)  # SEM1, SEM3
parser.add_argument('--noise_size', default=128, type=int)  # 64, 128
parser.add_argument('--smooth_param', default=0.23, type=float)
# SEM1(64): 0.18, 0.19 | SEM1(128): 0, 0.19, 0.23, 0.28
# SEM3(64): 0.19, 0.2 | SEM3(128): 0, 0.2, 0.3, 0.6
parser.add_argument('--mixed', default=True, type=bool)
parser.add_argument('--random_shift', default=False, type=bool)

# Transformations
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--mean', type=float, default=0.5)
parser.add_argument('--std', type=float, default=0.5)

args = parser.parse_args()

# Train Ne2Ne
train_VAE = TrainVAE(args=args)
train_VAE.train()

import argparse
from temp_train import TrainVAE

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

parser.add_argument('--img_size', default=256, type=int)
parser.add_argument('--pat_size', default=32, type=int)

# Transformations
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--mean', type=float, default=0.5)
parser.add_argument('--std', type=float, default=0.5)

args = parser.parse_args()

# Train Ne2Ne
train_VAE = TrainVAE(args=args)
train_VAE.train()

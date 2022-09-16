import argparse
import random
import time
from glob import glob

from models.VAE import VAE
from utils import *

# Arguments
parser = argparse.ArgumentParser(description='Test VAE')

parser.add_argument('--gpu_num', default=0, type=int)
parser.add_argument('--seed', default=100, type=int)
parser.add_argument('--exp_num', default=17, type=int)

# Training parameters
parser.add_argument('--n_epochs', default=200, type=int)
parser.add_argument('--n_downsample', default=2, type=int)
parser.add_argument('--n_upsample', default=2, type=int)

parser.add_argument('--img_size', default=256, type=int)
parser.add_argument('--pat_size', default=32, type=int)

# Test parameters
parser.add_argument('--n_samples', default=10, type=int)

# Transformations
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--mean', type=float, default=0.5)
parser.add_argument('--std', type=float, default=0.5)

opt = parser.parse_args()


def generate(args):
    device = torch.device('cuda:{}'.format(args.gpu_num))

    # Random Seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Model
    # model = tVAE(input_dim=1, norm='bn', device=device).to(device)
    model = VAE(device=device, n_downsample=args.n_downsample, n_upsample=args.n_upsample).to(device)
    model.load_state_dict(torch.load('./experiments/exp{}/checkpoints/{}epochs.pth'.format(args.exp_num, args.n_epochs), map_location=device))
    model.eval()

    # Directory
    out_size = args.noise_size * args.multiple
    save_dir = './results/{}_{}_{}'.format(args.dataset_name, args.smooth_param, out_size)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    z_channel = 64 * (2 ** args.n_downsample)
    z_size = int(256 / (2 ** args.n_downsample))

    with torch.no_grad():
        for i in range(args.n_samples):
            z = torch.randn(1, z_channel, z_size, z_size).to(device)
            img = model.decode(z)[0]
            img = denorm(img, mean=args.mean, std=args.std)
            img = tensor_to_numpy(img)
            img = np.clip(img, 0., 1.) * 255.
            cv2.imwrite(os.path.join(save_dir, '{}.png'.format(i+1)), img)


if __name__ == "__main__":
    generate(opt)

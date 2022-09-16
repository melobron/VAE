import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import json
import random
from tqdm import tqdm

from utils import *
from models.VAE import VAE
from temp_dataset import Pattern


class TrainVAE:
    def __init__(self, args):
        # Arguments
        self.args = args

        # Device
        self.gpu_num = args.gpu_num
        self.device = torch.device('cuda:{}'.format(self.gpu_num) if torch.cuda.is_available() else 'cpu')

        # Random Seeds
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        # Training Parameters
        self.n_epochs = args.n_epochs
        self.step_size = args.step_size
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.kld_weight = args.kld_weight

        # Transformation Parameters
        self.mean = args.mean
        self.std = args.std

        # Model Parameters
        self.n_downsample = args.n_downsample
        self.n_upsample = args.n_upsample

        # Dataset Parameters
        self.img_size = args.img_size
        self.pat_size = args.pat_size

        # Loss
        self.criterion_L1 = nn.L1Loss()

        # Transform
        transform = transforms.Compose(get_transforms(args))

        # Model
        self.model = VAE(device=self.device, n_downsample=self.n_downsample, n_upsample=self.n_upsample).to(self.device)
        self.model.apply(weights_init_('kaiming'))

        # Dataset
        self.train_dataset = Pattern(img_size=self.img_size, pat_size=self.pat_size, transform=transform)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=False)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))

        # Scheduler
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)

        # Directories
        self.exp_dir = make_exp_dir('./experiments/')['new_dir']
        self.exp_num = make_exp_dir('./experiments/')['new_dir_num']
        self.checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        self.result_path = os.path.join(self.exp_dir, 'results')

        # Tensorboard
        self.summary = SummaryWriter('runs/exp{}'.format(self.exp_num))

    def prepare(self):
        # Save Paths
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        # Save Argument file
        param_file = os.path.join(self.exp_dir, 'params.json')
        with open(param_file, mode='w') as f:
            json.dump(self.args.__dict__, f, indent=4)

    def train(self):
        print(self.device)
        self.prepare()

        for epoch in range(1, self.n_epochs + 1):
            with tqdm(self.train_dataloader, desc='Epoch {}'.format(epoch)) as tepoch:
                for batch, pattern in enumerate(tepoch):
                    self.model.train()
                    self.optimizer.zero_grad()

                    pattern = pattern.to(self.device)

                    mu, logvar, _ = self.model.encode(pattern)
                    z = self.model.reparameterize(mu, logvar)

                    fake = self.model.decode(z)

                    kld_loss = torch.mean(torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0))
                    rec_loss = self.criterion_L1(fake, pattern)

                    loss = rec_loss + self.kld_weight * kld_loss
                    loss.backward()
                    self.optimizer.step()

                    tepoch.set_postfix(kld_loss=kld_loss.item(), rec_loss=rec_loss.item(), total_loss=loss.item())
                    self.summary.add_scalar('kld_loss', kld_loss.item(), epoch)
                    self.summary.add_scalar('rec_loss', rec_loss.item(), epoch)
                    self.summary.add_scalar('total_loss', loss.item(), epoch)

            self.scheduler.step()

            # Checkpoints
            if epoch % 25 == 0 or epoch == self.n_epochs:
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, '{}epochs.pth'.format(epoch)))

            z_channel = 64 * (2 ** self.n_downsample)
            z_size = int(256 / (2 ** self.n_downsample))

            with torch.no_grad():
                self.model.eval()

                sample_z = torch.randn(1, z_channel, z_size, z_size).to(self.device)
                sample_imgs = self.model.decode(sample_z)
                sample_img = sample_imgs[0]
                sample_img = denorm(sample_img, mean=self.mean, std=self.std)
                sample_img = tensor_to_numpy(sample_img)
                sample_img = np.clip(sample_img, 0., 1.) * 255.
                cv2.imwrite(os.path.join(self.result_path, '{}epochs.png'.format(epoch)), sample_img)
        self.summary.close()











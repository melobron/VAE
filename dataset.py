import random

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils import *


class SEMNoise(Dataset):
    def __init__(self, noise_size, smooth_param, multiple, mixed=True, random_shift=True, dataset='SEM1', transform=None):
        super(SEMNoise, self).__init__()

        data_dir = os.path.join('../all_datasets/', dataset)
        self.noise_dir = os.path.join(data_dir, 'smooth_patch_{}'.format(noise_size), '{}'.format(smooth_param))
        self.noise_paths = sorted(make_dataset(self.noise_dir))

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

        self.noise_size = noise_size
        self.multiple = multiple
        self.random_shift = random_shift
        self.mixed = mixed

    def __getitem__(self, index):
        if self.mixed:
            noise_path = self.noise_paths[index]
            noise = cv2.imread(noise_path, cv2.IMREAD_GRAYSCALE) / 255.
            noise_tile = self.get_random_tile(self.noise_size, self.multiple, self.random_shift)
            noise, noise_tile = self.transform(noise), self.transform(noise_tile)
        else:
            noise_path = self.noise_paths[index]
            noise = cv2.imread(noise_path, cv2.IMREAD_GRAYSCALE) / 255.
            noise = self.transform(noise)
            noise_tile = self.get_noise_tile(noise, self.noise_size, self.multiple, self.random_shift)
        noise, noise_tile = noise.type(torch.FloatTensor), noise_tile.type(torch.FloatTensor)
        return {'noise': noise, 'noise_tile': noise_tile}

    def __len__(self):
        return len(self.noise_paths)

    def get_noise_tile(self, noise, noise_size, multiple, random_shift):
        noise_tile = torch.tile(noise, (multiple+1, multiple+1))
        noise_tile_size = noise_size * multiple
        if random_shift:
            start_h, start_w = random.randrange(noise_size), random.randrange(noise_size)
        else:
            start_h, start_w = 0, 0
        end_h, end_w = start_h + noise_tile_size, start_w + noise_tile_size
        noise_tile = noise_tile[:, start_h:end_h, start_w:end_w]
        return noise_tile

    def get_random_tile(self, noise_size, multiple, random_shift):
        noise_tile = np.zeros(shape=(noise_size*(multiple+1), noise_size*(multiple+1)))
        for i in range(multiple+1):
            for j in range(multiple+1):
                start_h, start_w = 128 * i, 128 * j
                end_h, end_w = 128 * (i+1), 128 * (j+1)
                rand_index = random.randrange(len(self.noise_paths))
                img = cv2.imread(self.noise_paths[rand_index], cv2.IMREAD_GRAYSCALE) / 255.
                noise_tile[start_h:end_h, start_w:end_w] = img
        noise_tile_size = noise_size * multiple
        if random_shift:
            start_h, start_w = random.randrange(noise_size), random.randrange(noise_size)
        else:
            start_h, start_w = 0, 0
        end_h, end_w = start_h + noise_tile_size, start_w + noise_tile_size
        noise_tile = noise_tile[start_h:end_h, start_w:end_w]
        return noise_tile



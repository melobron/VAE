import random

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils import *


class Pattern(Dataset):
    def __init__(self, img_size=512, pat_size=32, transform=None):
        super(Pattern, self).__init__()

        self.img_size = img_size
        self.pat_size = pat_size

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        pattern = self.generate_pattern(self.img_size, self.pat_size)
        pattern = self.transform(pattern)
        pattern = pattern.type(torch.FloatTensor)
        return pattern

    def __len__(self):
        return 1000

    def generate_pattern(self, img_size, pat_size):
        pattern = np.zeros(shape=(img_size, img_size))
        start_index = random.randrange(pat_size)
        for i in range(start_index, img_size, pat_size*2):
            pattern[:, i:i+pat_size] = 1
        return pattern

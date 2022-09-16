import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

import torch
import torch.nn.init as init
from torchvision.transforms import transforms


################################# Path & Directory #################################
def is_image_file(filename):
    extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.tif', '.TIF']
    return any(filename.endswith(extension) for extension in extensions)


def make_dataset(dir):
    img_paths = []
    assert os.path.isdir(dir), '{} is not a valid directory'.format(dir)

    for (root, dirs, files) in sorted(os.walk(dir)):
        for filename in files:
            if is_image_file(filename):
                img_paths.append(os.path.join(root, filename))
    return img_paths


def make_exp_dir(main_dir):
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)

    dirs = os.listdir(main_dir)
    dir_nums = []
    for dir in dirs:
        dir_num = int(dir[3:])
        dir_nums.append(dir_num)
    if len(dirs) == 0:
        new_dir_num = 1
    else:
        new_dir_num = max(dir_nums) + 1
    new_dir_name = 'exp{}'.format(new_dir_num)
    new_dir = os.path.join(main_dir, new_dir_name)
    return {'new_dir': new_dir, 'new_dir_num': new_dir_num}


################################# Transforms #################################
def get_transforms(args):
    transform_list = [transforms.ToTensor()]
    if args.normalize:
        transform_list.append(transforms.Normalize(mean=args.mean, std=args.std))
    return transform_list


def crop(img, patch_size):
    if img.ndim == 2:
        h, w = img.shape
        return img[h//2-patch_size//2:h//2+patch_size//2, w//2-patch_size//2:w//2+patch_size//2]
    elif img.ndim == 3:
        c, h, w = img.shape
        return img[:, h//2-patch_size//2:h//2+patch_size//2, w//2-patch_size//2:w//2+patch_size//2]
    else:
        raise NotImplementedError('Wrong image dim')


################################# ETC #################################
def denorm(tensor, mean=0.5, std=0.5, max_pixel=1.):
    return std*max_pixel*tensor + mean*max_pixel


def tensor_to_numpy(x):
    x = x.detach().cpu().numpy()
    if x.ndim == 4:
        return x.transpose((0, 2, 3, 1))
    elif x.ndim == 3:
        return x.transpose((1, 2, 0))
    elif x.ndim == 2:
        return x
    else:
        raise


def plot_tensors(tensor_list, title_list):
    numpy_list = [tensor_to_numpy(t) for t in tensor_list]
    fig = plt.figure(figsize=(4*len(numpy_list), 8))
    rows, cols = 1, len(numpy_list)
    for i in range(len(numpy_list)):
        plt.subplot(rows, cols, i+1)
        plt.imshow(numpy_list[i], cmap='gray')
        plt.title(title_list[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


def weights_init_(init_type='gaussian'):
   def init_fun(m):
       classname = m.__class__.__name__
       if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
           # print m.__class__.__name__
           if init_type == 'gaussian':
               init.normal_(m.weight.data, 0.0, 0.02)
           elif init_type == 'xavier':
               init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
           elif init_type == 'kaiming':
               init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
           elif init_type == 'orthogonal':
               init.orthogonal_(m.weight.data, gain=math.sqrt(2))
           elif init_type == 'default':
               pass
           else:
               assert 0, "Unsupported initialization: {}".format(init_type)
           if hasattr(m, 'bias') and m.bias is not None:
               init.constant_(m.bias.data, 0.0)
   return init_fun


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

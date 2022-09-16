import os
from glob import glob
import cv2
import random
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# gauss = 0.37 + np.random.randn(128, 128, 1) * 0.4
# img_paths = [
#     # './experiments/exp4/results/200epochs.png',
#     # './experiments/exp5/results/200epochs.png',
#     # './experiments/exp6/results/200epochs.png',
#     # './experiments/exp7/results/200epochs.png',
#     '../all_datasets/SEM1/smooth_patch_128/0.23/1.png',
#     './results/exp3_m1/1.png',
#     './results/exp3_m2/1.png',
#     './results/exp3_m3/1.png',
#     './results/exp3_m4/1.png',
#     './results/exp3_m5/1.png',
#     './results/exp3_m6/1.png'
# ]
#
# fig = plt.figure(figsize=(15, 10))
# rows, cols = 2, 4
#
# for index, img_path in enumerate(img_paths):
#     fig.add_subplot(rows, cols, index+2)
#     if index <= 1:
#         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)[:128, :128] / 255.
#     else:
#         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)[64:192, 64:192] / 255.
#     print(np.mean(img))
#     plt.imshow(img, cmap='gray')
#     if index == 0:
#         plt.title('sem')
#     # else:
#     #     plt.title('exp{}'.format(index+4))
#     plt.axis('off')
#
# fig.add_subplot(rows, cols, 1)
# img = 0.34 + np.random.randn(128, 128, 1)
# plt.imshow(img, cmap='gray')
# plt.title('gaussian')
# plt.axis('off')
#
# plt.tight_layout()
# plt.show()

img_size, pat_size = 256, 32
def gen_pattern():
    pattern = np.zeros(shape=(img_size, img_size))
    start_index = random.randrange(pat_size)
    for i in range(start_index, img_size, pat_size*2):
        pattern[:, i:i + pat_size] = 1
    return pattern

fig = plt.figure(figsize=(15, 10))
rows, cols = 1, 3

for i in range(1, 4):
    fig.add_subplot(rows, cols, i)
    img = gen_pattern()
    plt.imshow(img, cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()

# import libraries
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

warnings.filterwarnings("ignore")

# collecting images from DIR
path = './data/raw/CATS_DOGS/'
img_names = []

# for folder, subfolders, filenames in os.walk(path):
#     for img in filenames:
#         img_names.append(folder + '/' + img)

# print('Images: ', len(img_names))
# print(img_names[:5])

# image integrity checks + analysis
img_sizes = []
rejected = []

# for i, item in enumerate(img_names):
#     if i % 2400 == 0:
#         print(f'Checkpoint i: {i: 5}')

#     try:
#         with Image.open(item) as img:
#             img_sizes.append(img.size)
#     except:
#         rejected.append(item)

# print(f'Images: {len(img_sizes)}')
# print(f'Rejects: {len(rejected)}')

# df = pd.DataFrame(img_sizes)

# print('\nSummary statistics on Image widths')
# print(df[0].describe())

# print('\nSummary statistics on Image heights')
# print(df[1].describe())

# transformation test toy
dog = Image.open('./data/raw/CATS_DOGS/train/DOG/14.jpg')
print(dog.size)
# dog.show()

r, g, b = dog.getpixel((0,0))
print(r, g, b)

transform = transforms.Compose([
    transforms.ToTensor()
])
im = transform(dog)
print(im.shape)
# plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
# plt.show()

print(im[:, 0, 0])

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])
im = transform(dog)
print(im.shape)
# plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
# plt.show()

small_dog = Image.open('./data/raw/CATS_DOGS/train/DOG/11.jpg')
print(small_dog.size)
# small_dog.show()

im = transform(small_dog)
print(im.shape)
# plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
# plt.show()

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
im = transform(dog)
print(im.shape)
# plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
# plt.show()

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p = 1),
    transforms.ToTensor()
])
im = transform(dog)
print(im.shape)
# plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
# plt.show()

transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.ToTensor()
])
im = transform(dog)
print(im.shape)
# plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
# plt.show()

transform = transforms.Compose([    # scaling with two dimension params
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
im = transform(dog)
print(im.shape)
# plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
# plt.show()

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p = 1),
    transforms.RandomRotation(30),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
im = transform(dog)
print(im.shape)
# plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
# plt.show()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
im = transform(dog)
print(im.shape)
# plt.imshow(np.transpose(im.numpy(), ( 1, 2 , 0)))
# plt.show()

print(im[:, 0, 0])

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
im_inv = inv_normalize(im)
plt.figure(figsize = (12, 4))
plt.imshow(np.transpose(im_inv.numpy(), (1, 2, 0)))
plt.show()

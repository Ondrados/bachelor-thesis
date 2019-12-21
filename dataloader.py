import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import glob
import os
from skimage.io import imread

from torch.utils.data import Dataset


class DataLoader(Dataset):
    def __init__(self, split="stage1_train", path_to_data='/Users/ondra/Dev/Personal/cnn-cells/data-science-bowl-2018'):
        self.split = split
        self.path = path_to_data + '/' + split

        self.id_list = os.listdir(self.path)
        self.image_list = []
        self.mask_list = []

#       set_trace()
        for id in self.id_list:
            images = glob.glob(self.path + '/' + id + '/images/*png')
            masks = glob.glob(self.path + '/' + id + '/masks/*png')
            self.image_list.extend(images)
            self.mask_list.extend(masks)

        self.num_of_imgs = len(self.id_list)

    def __len__(self):
        return self.num_of_imgs

    def __getitem__(self, index):
        # zpracovat zde
        #       set_trace()
        img = imread(self.file_list[index])
        img = torch.Tensor(img.astype(np.float32) / 255 - 0.5)
        lbl = torch.Tensor(np.array(self.lbls[index]).astype(np.float32))
        return img, lbl
    # load and preprocess one image - with number index
    # torchvision.transforms  contains several preprocessing functions for images


loader = DataLoader(split='stage1_train')
trainloader = data.DataLoader(
    loader, batch_size=2, num_workers=0, shuffle=True, drop_last=True)


for it, (batch, lbls) in enumerate(trainloader):  # you can iterate over dataset (one epoch)
    print(batch)
    print(batch.size())
    print(lbls)
    plt.imshow(batch[0, :, :].detach().cpu().numpy())
    break

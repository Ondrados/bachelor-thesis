import os
import glob
import numpy as np
import torch

from skimage.io import imread
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, split="stage1_train", path_to_data='/Users/ondra/Dev/Personal/cnn-cells/data-science-bowl-2018'):
        self.split = split
        self.path = path_to_data + '/' + split

        self.id_list = os.listdir(self.path)
        self.image_list = []
        self.mask_list = []

        for id in self.id_list:
            images = glob.glob(self.path + '/' + id + '/images/*png')
            masks = glob.glob(self.path + '/' + id + '/masks/*png')
            self.image_list.extend(images)
            self.mask_list.append(masks)

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        im = imread(self.image_list[index])
        msk = self.combine_masks(self.mask_list[index])
        # pre process transforming to tensor
        image = torch.Tensor(im.astype(np.float32))
        mask = torch.Tensor(msk.astype(np.float32))
        return image, mask

    def combine_masks(self, mask_paths):
        comb_mask = None
        for path in mask_paths:
            mask = imread(path)
            if comb_mask is None:
                comb_mask = np.zeros_like(mask)
            comb_mask += mask
        return comb_mask

    def pre_process(self):
        # pre processing
        pass

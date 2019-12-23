import os
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use('GTKAgg')
from PIL import Image
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms as T



class MyDataset(Dataset):
    def __init__(self, transforms=None, split="stage1_train", path_to_data='/home/ubmi/Documents/cnn-cells/cnn-cells/data-science-bowl-2018'):
        self.split = split
        self.path = path_to_data + '/' + split

        self.path_id_list = glob.glob(os.path.join(self.path, '*'))
        self.image_list = []
        self.mask_list = []

        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = T.Compose([
                T.CenterCrop(256),
                T.Grayscale(num_output_channels=1),
                T.ToTensor()
            ])

        for path_id in self.path_id_list:
            images = glob.glob(path_id + '/images/*png')
            masks = glob.glob(path_id + '/masks/*png')
            self.image_list.extend(images)
            self.mask_list.append(masks)

    def __len__(self):
        return len(self.path_id_list)

    def __getitem__(self, index):
        # im = resize(rgb2gray(imread(self.image_list[index])),(256, 256))
        # im = rgb2gray(imread(self.image_list[index]))
        image = Image.open(self.image_list[index])
        # msk = resize(self.combine_masks(self.mask_list[index]),(256, 256))
        mask = self.combine_masks(self.mask_list[index])
        # pre process before transforming to tensor
        # image = torch.Tensor(im.astype(np.float32))
        # mask = torch.Tensor(msk.astype(np.float32))
        image = self.transforms(image)
        mask = self.transforms(mask)
        return image, mask

    def combine_masks(self, mask_paths):
        comb_mask = None
        for path in mask_paths:
            #mask = imread(path)
            mask = Image.open(path)
            if comb_mask is None:
                comb_mask = np.zeros_like(mask)
            comb_mask += mask
        return Image.fromarray(comb_mask)

    def pre_process(self):
        # pre processing
        pass

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    dataset = MyDataset(split='stage1_train', path_to_data = '/home/ubmi/Documents/cnn-cells/cnn-cells/data-science-bowl-2018')
    trainloader = DataLoader(dataset,batch_size=1, num_workers=0, shuffle=True, drop_last=True)

    for i, data in enumerate(trainloader):
        inputs, masks = data[0].to(device=device), data[1].to(device=device)
        fig = plt.figure(figsize=(1, 2))
        fig.add_subplot(1, 2, 1)
        plt.imshow(inputs[0,0,:,:].detach().cpu().numpy(), cmap="gray")
        fig.add_subplot(1, 2, 2)
        plt.imshow(masks[0,0,:,:].detach().cpu().numpy(), cmap="gray")

        plt.show()
        break
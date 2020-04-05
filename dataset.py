import os
import glob
import numpy as np
import torch
import matplotlib
from PIL import Image
from skimage import draw
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
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
        image = Image.open(self.image_list[index])
        mask = self.combine_masks(self.mask_list[index])
        # # make transforms
        # fig = plt.figure()
        # # fig.add_subplot(1, 2, 1)
        # plt.title("Image")
        # plt.imshow(image)
        # # fig.add_subplot(1, 2, 2)
        # # plt.title("Mask")
        # # plt.imshow(mask, cmap="gray")
        # # plt.show()
        # plt.savefig('images/dataset-6.png')
        image = self.transforms(image)
        mask = self.transforms(mask)

        plt.show(block=True)
        return image, mask

    def combine_masks(self, mask_paths):
        comb_mask = None
        for path in mask_paths:
            # mask = Image.open(path)
            mask = imread(path)
            count = (mask == 255).sum()
            y, x = np.argwhere(mask == 255).sum(0) / count
            if comb_mask is None:
                # comb_mask = np.zeros_like(mask)
                comb_mask = np.zeros_like(mask)
            # comb_mask += mask
            rr, cc = draw.circle(y, x, radius=3)
            try:
                comb_mask[rr, cc] = 255
            except IndexError:
                pass
            blurred = gaussian_filter(comb_mask, sigma=1)
        # fig = plt.figure()
        # fig.add_subplot(1, 2, 1)
        # plt.imshow(comb_mask,cmap="gray")
        # fig.add_subplot(1, 2, 2)
        # plt.imshow(comb_mask,cmap="gray")
        # plt.show(block=True)

        return Image.fromarray(blurred)

    def pre_process(self):
        # pre processing
        pass

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    dataset = MyDataset(split='stage1_train', path_to_data = '/Users/ondra/Dev/Personal/cnn-cells/data-science-bowl-2018')
    trainloader = DataLoader(dataset,batch_size=1, num_workers=0, shuffle=True, drop_last=True)

    for i, data in enumerate(trainloader):
        inputs, masks = data[0].to(device=device), data[1].to(device=device)
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.title("Image")
        plt.imshow(inputs[0,0,:,:].detach().cpu().numpy(), cmap="gray")
        fig.add_subplot(1, 2, 2)
        plt.imshow(masks[0,0,:,:].detach().cpu().numpy(), cmap="gray")
        plt.title("Mask")

        plt.show()
        # plt.savefig('images/image_mask-99.png')
        plt.close("all")
        break
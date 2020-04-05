import os
import glob
import numpy as np
import torch
import matplotlib
from PIL import Image, ImageDraw
from skimage import draw
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from settings import BASE_DIR
from utils import transforms as my_T

dataset_path = os.path.join(BASE_DIR, 'data-science-bowl-2018')


def get_transform(train=False):
    transforms = []
    transforms.append(my_T.ToTensor())
    if train:
        # TODO: add train transforms
        pass
    return T.Compose(transforms)




class MyDataset(Dataset):
    def __init__(self, transforms=None, split="stage1_train", path=dataset_path):
        self.split = split
        self.path = path + '/' + split

        self.transforms = transforms

        self.path_id_list = glob.glob(os.path.join(self.path, '*'))
        self.id_list = []
        self.image_list = []
        self.mask_list = []


        for path_id in self.path_id_list:
            images = glob.glob(path_id + '/images/*png')
            masks = glob.glob(path_id + '/masks/*png')
            self.id_list.append(os.path.basename(path_id))
            self.image_list.extend(images)
            self.mask_list.append(masks)

    def __len__(self):
        return len(self.path_id_list)

    def __getitem__(self, index):
        # print(f"ID: {self.id[index]}")
        # print(f"Image: {self.image_list[index]}")
        # print(f"Masks: {self.mask_list[index]}")
        image = Image.open(self.image_list[index])
        # mask = self.combine_masks(self.mask_list[index])
        boxes = self.mask_to_bbox(self.mask_list[index])
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
        target = {}
        target["id"] = self.id_list[index]
        target["boxes"] = boxes
        # target["mask"] = mask

        if self.transforms is not None:
            sample = {'image': image, 'target': target}
            sample = self.transforms(sample)

        plt.show(block=True)
        return image, target

    def mask_to_bbox(self, mask_paths):
        boxes = []
        for path in mask_paths:
            mask = Image.open(path)
            mask = np.array(mask)
            pos = np.where(mask)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        return boxes

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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    dataset = MyDataset(split='stage1_train', transforms=get_transform())
    trainloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True, drop_last=True)

    image, target = next(iter(dataset))

    boxes = target["boxes"]

    draw = ImageDraw.Draw(image)
    for box in boxes:
        x0, y0, x1, y1 = box
        draw.rectangle([(x0, y0), (x1, y1)], outline=(255, 0, 255))

    image.show(title=target["id"])

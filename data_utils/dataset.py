import os
import glob
import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage import draw
from skimage.io import imread
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from conf.settings import BASE_DIR
from data_utils import transforms as my_T

dataset_path = os.path.join(BASE_DIR, 'data-science-bowl-2018')


def my_collate(batch):
    image = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return image, targets


def get_transforms(train=False, rescale_size=(256, 256), yolo=False):
    transforms = []
    if train:
        transforms.append(my_T.Rescale(rescale_size, yolo))
        transforms.append(my_T.Normalize())
    transforms.append(my_T.ToTensor())
    return my_T.Compose(transforms)


class MyDataset(Dataset):
    def __init__(self, transforms=None, split="stage1_train", path=dataset_path, model=None):
        self.split = split
        self.path = path + '/' + split

        self.transforms = transforms
        self.model = model

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

    def __getitem__(self, index, unet=False):
        # TODO: make full unet integration
        if self.model == "unet":
            image = Image.open(self.image_list[index])
            mask = self.combine_masks(self.mask_list[index])
            self.transforms = T.Compose([
                T.CenterCrop(256),
                T.Grayscale(num_output_channels=1),
                T.ToTensor()
            ])
            image = self.transforms(image)
            mask = self.transforms(mask)
            return image, mask

        image = np.array(Image.open(self.image_list[index]), dtype=np.uint8)
        image = image[:, :, :3]  # remove alpha channel
        boxes, labels = self.mask_to_bbox(self.mask_list[index])
        targets = {
            'boxes': torch.FloatTensor(boxes),
            'labels': torch.LongTensor(labels),
            'name': self.id_list[index]
        }

        if self.transforms is not None:
            image, targets = self.transforms(image, targets)

        return image, targets

    def mask_to_bbox(self, mask_paths):
        boxes = []
        labels = []
        for path in mask_paths:
            mask = Image.open(path)
            mask = np.array(mask)
            pos = np.where(mask)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if xmin != xmax and ymin != ymax:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(1)  # every mask is cell
        return boxes, labels

    def combine_masks(self, mask_paths):
        comb_mask = None
        comb_mask_def = None
        for path in mask_paths:
            mask = Image.open(path)
            if comb_mask_def is None:
                comb_mask_def = np.zeros_like(mask)
            comb_mask_def += mask
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


def get_test_transforms(rescale_size=(416, 416)):
    transforms = [my_T.TestRescale(rescale_size), my_T.Normalize(), my_T.ToTensor()]
    return my_T.Compose(transforms)


class MyTestDataset(Dataset):
    def __init__(self, transforms=get_test_transforms(), split="stage1_test", path=dataset_path, model=None):
        self.split = split
        self.path = path + '/' + split

        self.transforms = transforms
        self.model = model

        self.path_id_list = glob.glob(os.path.join(self.path, '*'))
        self.id_list = []
        self.image_list = []

        for path_id in self.path_id_list:
            images = glob.glob(path_id + '/images/*png')
            self.image_list.extend(images)
            self.id_list.append(os.path.basename(path_id))

    def __len__(self):
        return len(self.path_id_list)

    def __getitem__(self, index):
        # TODO: make full unet integration
        if self.model == "unet":
            image = Image.open(self.image_list[index])
            self.transforms2 = T.Compose([
                T.CenterCrop(256),
                T.Grayscale(num_output_channels=1),
                T.ToTensor()
            ])
            image = self.transforms2(image)
            return image
        image = np.array(Image.open(self.image_list[index]), dtype=np.uint8)
        image = image[:, :, :3]  # remove alpha channel
        targets = {
            'name': self.id_list[index]
        }
        if self.transforms:
            image, targets = self.transforms(image, targets)

        return image, targets


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    model = "faster"

    if model == "yolo":
        dataset = MyDataset(split='stage1_train',
                            transforms=get_transforms(train=True, rescale_size=(416, 416), yolo=True))
        trainloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True, collate_fn=my_collate)
        it = iter(trainloader)
        image, targets = next(it)
        image = image[0].to(device=device)
        targets = [{
            "boxes": targets[0]["boxes"].to(device=device),
            "labels": targets[0]["labels"].to(device=device),
            "name": targets[0]["name"]
        }]
        _, _, h, w = image.shape
        image = Image.fromarray(image.numpy()[0, 0, :, :])

        if image.mode != "RGB":
            image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        for box in targets[0]["boxes"]:
            cls, xcnt, ycnt, width, height = box
            width = width * w
            height = height * h
            xcnt = xcnt * w
            ycnt = ycnt * h
            x0 = xcnt - (width / 2)
            x1 = xcnt + (width / 2)
            y0 = ycnt - (height / 2)
            y1 = ycnt + (height / 2)
            draw.rectangle([(x0, y0), (x1, y1)], outline=(255, 0, 255))

        image.show(title=targets[0]["name"])
    elif model == "faster":
        dataset = MyDataset(split='stage1_train', transforms=get_transforms(train=True))
        trainloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False, collate_fn=my_collate)
        it = iter(trainloader)
        image, targets = next(it)
        image = image[0].to(device=device)
        targets = [{
            "boxes": targets[0]["boxes"].to(device=device),
            "labels": targets[0]["labels"].to(device=device),
            "name": targets[0]["name"]
        }]

        image = Image.fromarray(image.numpy()[0, 0, :, :])
        if image.mode != "RGB":
            image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        for box in targets[0]["boxes"]:
            x0, y0, x1, y1 = box
            draw.rectangle([(x0, y0), (x1, y1)], outline=(255, 0, 255))

        # image.show(title=targets[0]["name"])
        plt.imshow(image)
        plt.show()
    else:
        dataset = MyDataset(split='stage1_train', model=model)

        inputs, masks = next(iter(dataset))
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.title("Obraz")
        plt.imshow(inputs[0,:,:].detach().cpu().numpy(), cmap="gray")
        fig.add_subplot(1, 2, 2)
        plt.imshow(masks[0,:,:].detach().cpu().numpy(), cmap="gray")
        plt.title("Maska")

        plt.show()

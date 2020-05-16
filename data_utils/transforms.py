import torch
import cv2
import numpy as np


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size, yolo=False):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.yolo = yolo

    def __call__(self, image, targets):

        h, w = image.shape[:2]
        if (h, w) == self.output_size:
            return image, targets

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # image = np.resize(image, (new_h, new_w, 3))
        image = cv2.resize(image, dsize=(new_h, new_w), interpolation=cv2.INTER_CUBIC)

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        ratio_height = new_h / h
        ratio_width = new_w / w

        xmin, ymin, xmax, ymax = targets["boxes"].unbind(1)

        xmin = xmin * ratio_width
        xmax = xmax * ratio_width
        ymin = ymin * ratio_height
        ymax = ymax * ratio_height

        if self.yolo:
            cls = torch.zeros(len(targets["boxes"]), dtype=torch.float)
            xcnt = (xmin + ((xmax - xmin) / 2)) / new_w
            ycnt = (ymin + ((ymax - ymin) / 2)) / new_h
            width = (xmax - xmin) / new_w
            height = (ymax - ymin) / new_h
            targets["boxes"] = torch.stack((cls, xcnt, ycnt, width, height), dim=1)
        else:
            targets["boxes"] = torch.stack((xmin, ymin, xmax, ymax), dim=1)

        return image, targets


class TestRescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image, targets):

        h, w = image.shape[:2]
        if (h, w) == self.output_size:
            return image, targets

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # image = np.resize(image, (new_h, new_w, 3))
        image = cv2.resize(image, dsize=(new_h, new_w), interpolation=cv2.INTER_CUBIC)

        return image, targets


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        landmarks = landmarks[0]
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top, left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image, targets):

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image[None, :, :, :]).float()
        return image, targets


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, targets):
        for t in self.transforms:
            image, targets = t(image, targets)
        return image, targets

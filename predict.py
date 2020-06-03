import os
import math
import torch
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import random_split, DataLoader
from matplotlib import pyplot as plt
from data_utils import MyTestDataset, get_test_transforms, my_collate
from conf.settings import BASE_DIR

from faster_rcnn.predict import predict as faster_predict
from yolo_v3.predict import predict as yolo_predict
from unet.predict import predict as unet_predict

models_path = os.path.join(BASE_DIR, "models")
images_path = os.path.join(BASE_DIR, "images")

if __name__ == "__main__":
    torch.manual_seed(0)

    from faster_rcnn.models import model as faster
    from yolo_v3.models import Darknet
    from unet.models import UNet

    faster_name = "faster_rcnn_7_30.pt"
    yolo_name = "yolo_v3_4_20.pt"
    unet_name = "unet_2_15.pt"

    split = "stage1_test"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    print(f"Loading {faster_name}")
    faster.load_state_dict(torch.load(os.path.join(models_path, "faster_rcnn_7_30.pt"), map_location=device))
    faster.to(device=device)
    dataset = MyTestDataset(split=split, transforms=get_test_transforms(rescale_size=(256, 256)))
    faster_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

    print(f"Loading {yolo_name}")
    yolo = Darknet(os.path.join(BASE_DIR, "yolo_v3/config/yolov3-custom.cfg"))
    yolo.load_state_dict(torch.load(os.path.join(models_path, yolo_name), map_location=device))
    yolo.to(device=device)
    dataset = MyTestDataset(split=split, transforms=get_test_transforms(rescale_size=(416, 416)))
    yolo_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

    print(f"Loading {unet_name}")
    unet = UNet(n_channels=1, n_classes=1)
    unet.load_state_dict(torch.load(os.path.join(models_path, unet_name), map_location=device))
    unet.to(device=device)
    dataset = MyTestDataset(split=split, model="unet")
    unet_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

    for i, ((f_im, f_tar), (y_im, y_tar), (u_im, u_tar)) in enumerate(zip(faster_loader, yolo_loader, unet_loader)):

        f_image, f_x, f_y = faster_predict(faster, image=f_im)
        y_image, y_x, y_y = yolo_predict(yolo, image=y_im)
        u_image, u_x, u_y = unet_predict(unet, image=u_im)

        fig = plt.figure(dpi=300)
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(f_image, cmap="gray")
        ax1.plot(f_x, f_y, 'r+', linewidth=3, markersize=12)
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(y_image, cmap="gray")
        ax2.plot(y_x, y_y, 'r+', linewidth=3, markersize=12)
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(u_image, cmap="gray")
        ax3.plot(u_x, u_y, 'r+', linewidth=3, markersize=12)
        plt.show()

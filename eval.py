import os
import math
import torch
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import random_split, DataLoader
from matplotlib import pyplot as plt
from data_utils import MyDataset, get_transforms, my_collate
from conf.settings import BASE_DIR

from faster_rcnn.eval import evaluate as faster_evaluate
from yolo_v3.eval import evaluate as yolo_evaluate
from unet.eval import evaluate as unet_evaluate

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

    split = "stage1_train"

    os.makedirs(os.path.join(images_path, "plots"), exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    print(f"Loading {faster_name}")
    faster.load_state_dict(torch.load(os.path.join(models_path, faster_name), map_location=device))
    faster.to(device=device)
    dataset = MyDataset(split=split, transforms=get_transforms(train=True, rescale_size=(256, 256)))
    _, f_evalset = random_split(dataset, [600, 70])
    faster_eval_loader = DataLoader(f_evalset, batch_size=1, num_workers=0, shuffle=False, collate_fn=my_collate)
    f_precision, f_recall, f_dice, f_dice_vec = faster_evaluate(faster, faster_eval_loader, dist_threshold=3)
    print(f"{faster_name}, precision: {f_precision}, recall: {f_recall}, dice: {f_dice}")
    print(f_dice_vec)

    print(f"Loading {yolo_name}")
    yolo = Darknet(os.path.join(BASE_DIR, "yolo_v3/config/yolov3-custom.cfg"))
    yolo.load_state_dict(torch.load(os.path.join(models_path, yolo_name), map_location=device))
    yolo.to(device=device)
    dataset = MyDataset(split=split, transforms=get_transforms(train=True, rescale_size=(416, 416), yolo=True))
    _, y_evalset = random_split(dataset, [600, 70])
    yolo_eval_loader = DataLoader(y_evalset, batch_size=1, num_workers=0, shuffle=False, collate_fn=my_collate)
    y_precision, y_recall, y_dice, y_dice_vec = yolo_evaluate(yolo, yolo_eval_loader, dist_threshold=6)
    print(f"{yolo_name}, precision: {y_precision}, recall: {y_recall}, dice: {y_dice}")
    print(y_dice_vec)

    print(f"Loading {unet_name}")
    unet = UNet(n_channels=1, n_classes=1)
    unet.load_state_dict(torch.load(os.path.join(models_path, unet_name), map_location=device))
    unet.to(device=device)
    dataset = MyDataset(split=split, model="unet")
    _, u_evalset = random_split(dataset, [600, 70])
    unet_eval_loader = DataLoader(u_evalset, batch_size=1, num_workers=1, shuffle=True, drop_last=True)
    u_precision, u_recall, u_dice, u_dice_vec = unet_evaluate(unet, unet_eval_loader, dist_threshold=3)
    print(f"{unet_name}, precision: {u_precision}, recall: {u_recall}, dice: {u_dice}")
    print(u_dice_vec)

    fig, ax = plt.subplots()
    ax.set_title('Multiple Samples with Different sizes')
    ax.boxplot([f_dice_vec, y_dice_vec, u_dice_vec])
    plt.savefig(os.path.join(images_path, f"plots/boxplot.png"), dpi=200)


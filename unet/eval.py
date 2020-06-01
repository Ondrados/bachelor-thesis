import os
import math
import torch
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import random_split, DataLoader
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
from skimage.morphology import extrema
from skimage.measure import label
from skimage import color
from models import UNet

from data_utils import MyDataset

from conf.settings import BASE_DIR

models_path = os.path.join(BASE_DIR, "models")
images_path = os.path.join(BASE_DIR, "images")


def evaluate(model, eval_loader, dist_threshold=3):
    runnning_dice_vec = 0
    runnning_prec_vec = 0
    runnning_rec_vec = 0
    model.eval()
    for i, (image, mask, name, x, y) in enumerate(eval_loader):

        image = image[0].to(device=device)[None, :, :, :]

        with torch.no_grad():
            outputs = model(image)

        out = outputs[0, 0, :, :].numpy()
        array = np.where(out < 0.3, 0, out)

        h = 0.3
        h_maxima = extrema.h_maxima(array, h)
        label_h_maxima = label(h_maxima)

        coordinates = peak_local_max(label_h_maxima, indices=True)

        # fig = plt.figure(dpi=300)
        # ax1 = fig.add_subplot(1, 3, 1)
        # ax1.imshow(image[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
        # ax1.plot(coordinates[:, 1], coordinates[:, 0], 'r+', markersize=10)
        # ax2 = fig.add_subplot(1, 3, 2)
        # ax2.imshow(outputs[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
        # ax3 = fig.add_subplot(1, 3, 3)
        # # ax3.imshow(inputs[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
        # # ax3.imshow(outputs[0, 0, :, :].detach().cpu().numpy(), cmap="jet", alpha=0.3)
        # ax3.imshow(array)
        # plt.show()

        gt_x = x
        gt_y = y

        pred_x = coordinates[:, 1]
        pred_y = coordinates[:, 0]

        # fig = plt.figure(dpi=300)
        # ax1 = fig.add_subplot(1, 1, 1)
        # ax1.imshow(image[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
        # ax1.plot(gt_x, gt_y, 'g+', linewidth=3, markersize=12)
        # ax1.plot(pred_x, pred_y, 'm+', linewidth=3, markersize=12)
        # plt.show()

        dist_matrix = np.zeros((len(gt_x), len(pred_x)))
        for row, (g_x, g_y) in enumerate(zip(gt_x, gt_y)):
            for col, (p_x, p_y) in enumerate(zip(pred_x, pred_y)):
                x = abs(g_x - p_x)
                y = abs(g_y - p_y)
                dist_matrix[row, col] = math.sqrt((x*x)+(y*y))

        min_dists = np.amin(dist_matrix, axis=0)

        tp = 0
        fp = 0
        for dist in min_dists:
            if dist <= dist_threshold:
                tp += 1
            else:
                fp += 1
        tp = len(gt_x) if tp > len(gt_x) else tp
        fn = len(gt_x) - tp
        if (tp + fp) == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        dice = (2 * tp) / (2 * tp + fp + fn)
        runnning_dice_vec += dice
        runnning_prec_vec += precision
        runnning_rec_vec += recall

        print(f"{i}, TP: {tp}, FP: {fp}, FN: {fn}, precision: {precision}, recall: {recall}, dice: {dice}")
        # print(f"Iteration: {i} of {len(eval_loader)}, image: {name}")

    # dice_vec.append(runnning_dice_vec / len(eval_loader))
    # prec_vec.append(runnning_prec_vec / len(eval_loader))
    # rec_vec.append(runnning_rec_vec / len(eval_loader))

    prec_result = runnning_prec_vec / len(eval_loader)
    rec_result = runnning_rec_vec / len(eval_loader)
    dice_result = runnning_dice_vec / len(eval_loader)

    return prec_result, rec_result, dice_result


if __name__ == "__main__":
    attempt = 2

    model_name = "unet_2_15.pt"

    os.makedirs(models_path, exist_ok=True)
    os.makedirs(os.path.join(images_path, f"faster_rcnn/{attempt}/images"), exist_ok=True)
    os.makedirs(os.path.join(images_path, f"faster_rcnn/{attempt}/plots"), exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Running on {device}, using {model_name}")

    model = UNet(n_channels=1, n_classes=1)
    model.load_state_dict(torch.load(os.path.join(models_path, model_name), map_location=device))

    split = "stage1_train"
    dataset = MyDataset(split=split, model="unet")
    trainset, evalset = random_split(dataset, [600, 70])

    eval_loader = DataLoader(evalset, batch_size=1, num_workers=1, shuffle=True, drop_last=True)

    precision, recall, dice = evaluate(model, eval_loader, dist_threshold=3)

    print(f"Done, precision: {precision}, recall: {recall}, dice: {dice}")

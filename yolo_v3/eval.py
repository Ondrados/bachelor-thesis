import os
import math
import torch
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import random_split, DataLoader
from matplotlib import pyplot as plt
from models import Darknet

from data_utils import MyDataset, get_transforms, my_collate
from utils import non_max_suppression, rescale_boxes

from conf.settings import BASE_DIR

models_path = os.path.join(BASE_DIR, "models")
images_path = os.path.join(BASE_DIR, "images")


def evaluate(model, eval_loader, dist_threshold=3):
    runnning_dice_vec = 0
    runnning_prec_vec = 0
    runnning_rec_vec = 0
    model.eval()
    for i, (image, targets) in enumerate(eval_loader):

        name = targets[0]["name"]
        image = image[0].to(device=device)
        targets = [{
            "boxes": targets[0]["boxes"].to(device=device),
            "name": name
        }]

        with torch.no_grad():
            outputs = model(image)
        outputs = non_max_suppression(outputs, conf_thres=0.5)
        boxes = outputs[0][:, 0:4]
        _, _, h, w = image.shape
        image_copy = Image.fromarray(image.cpu().numpy()[0, 0, :, :])
        if image_copy.mode != "RGB":
            image_copy = image_copy.convert("RGB")
        # draw = ImageDraw.Draw(image_copy)
        # for box in targets[0]["boxes"]:
        #     cls, xcnt, ycnt, width, height = box
        #     width = width * w
        #     height = height * h
        #     xcnt = xcnt * w
        #     ycnt = ycnt * h
        #     x0 = xcnt - (width / 2)
        #     x1 = xcnt + (width / 2)
        #     y0 = ycnt - (height / 2)
        #     y1 = ycnt + (height / 2)
        #     draw.rectangle([(x0, y0), (x1, y1)], outline=(0, 255, 0))
        # for box in boxes:
        #     x0, y0, x1, y1 = box
        #     draw.rectangle([(x0, y0), (x1, y1)], outline=(255, 0, 255))
        # # image_copy.show()
        # # image_copy.save(os.path.join(images_path, f"faster_rcnn/{attempt}/images/{name}.png"))
        # plt.imshow(image_copy)
        # plt.show()

        gt_x = []
        gt_y = []
        for box in targets[0]["boxes"]:
            cls, xcnt, ycnt, width, height = box
            gt_x.append(xcnt.tolist() * 416)
            gt_y.append(ycnt.tolist() * 416)

        pred_x = []
        pred_y = []
        for box in boxes:
            x0, y0, x1, y1 = box
            x = ((x0 + x1) / 2).tolist()
            y = ((y0 + y1) / 2).tolist()
            pred_x.append(x)
            pred_y.append(y)

        # fig = plt.figure(dpi=300)
        # ax1 = fig.add_subplot(1, 1, 1)
        # ax1.imshow(image_copy)
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
    attempt = 4

    model_name = "yolo_v3_4_20.pt"

    os.makedirs(models_path, exist_ok=True)
    os.makedirs(os.path.join(images_path, f"faster_rcnn/{attempt}/images"), exist_ok=True)
    os.makedirs(os.path.join(images_path, f"faster_rcnn/{attempt}/plots"), exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Running on {device}, using {model_name}")

    model = Darknet(os.path.join(BASE_DIR, "yolo_v3/config/yolov3-custom.cfg")).to(device)
    model.load_state_dict(torch.load(os.path.join(models_path, model_name), map_location=device))

    split = "stage1_train"
    dataset = MyDataset(split=split, transforms=get_transforms(train=True, rescale_size=(416, 416), yolo=True))
    trainset, evalset = random_split(dataset, [600, 70])

    eval_loader = DataLoader(evalset, batch_size=1, num_workers=0, shuffle=False, collate_fn=my_collate)

    precision, recall, dice = evaluate(model, eval_loader, dist_threshold=6)

    print(f"Done, precision: {precision}, recall: {recall}, dice: {dice}")

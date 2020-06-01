import os
import math
import torch
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import random_split, DataLoader
from matplotlib import pyplot as plt
from faster_rcnn.models import model

from data_utils import MyDataset, get_transforms, my_collate

from conf.settings import BASE_DIR

models_path = os.path.join(BASE_DIR, "models")
images_path = os.path.join(BASE_DIR, "images")


def evaluate(model, eval_loader, dist_threshold=3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    runnning_dice_vec = 0
    runnning_prec_vec = 0
    runnning_rec_vec = 0
    dice_vec = []
    model.eval()
    for i, (image, targets) in enumerate(eval_loader):

        name = targets[0]["name"]
        image = image[0].to(device=device)
        targets = [{
            "boxes": targets[0]["boxes"].to(device=device),
            "labels": targets[0]["labels"].to(device=device),
            "name": name
        }]

        with torch.no_grad():
            predictions = model(image)

        image_copy = Image.fromarray(image.cpu().numpy()[0, 0, :, :])
        if image_copy.mode != "RGB":
            image_copy = image_copy.convert("RGB")
        # draw = ImageDraw.Draw(image_copy)
        # for box in targets[0]["boxes"]:
        #     x0, y0, x1, y1 = box
        #     # draw.rectangle([(x0, y0), (x1, y1)], outline=(0, 255, 0))
        # for box, score in zip(predictions[0]["boxes"], predictions[0]["scores"]):
        #     if score > 0.5:
        #         x0, y0, x1, y1 = box
        #         # draw.rectangle([(x0, y0), (x1, y1)], outline=(255, 0, 255))
        # # image_copy.show()
        # # image_copy.save(os.path.join(images_path, f"faster_rcnn/{attempt}/images/{name}.png"))
        # plt.imshow(image_copy)
        # plt.show()

        gt_x = []
        gt_y = []
        for box in targets[0]["boxes"]:
            x0, y0, x1, y1 = box
            x = ((x0 + x1) / 2).tolist()
            y = ((y0 + y1) / 2).tolist()
            gt_x.append(x)
            gt_y.append(y)

        pred_x = []
        pred_y = []
        for box, score in zip(predictions[0]["boxes"], predictions[0]["scores"]):
            if score > 0.5:
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

        dice_vec.append(dice)

        print(f"{i}, TP: {tp}, FP: {fp}, FN: {fn}, precision: {precision}, recall: {recall}, dice: {dice}")
        # print(f"Iteration: {i} of {len(eval_loader)}, image: {name}")

    # dice_vec.append(runnning_dice_vec / len(eval_loader))
    # prec_vec.append(runnning_prec_vec / len(eval_loader))
    # rec_vec.append(runnning_rec_vec / len(eval_loader))

    prec_result = runnning_prec_vec / len(eval_loader)
    rec_result = runnning_rec_vec / len(eval_loader)
    dice_result = runnning_dice_vec / len(eval_loader)

    return prec_result, rec_result, dice_result, dice_vec


if __name__ == "__main__":
    from models import model

    attempt = 7
    model_name = "faster_rcnn_7_30.pt"

    os.makedirs(models_path, exist_ok=True)
    os.makedirs(os.path.join(images_path, f"faster_rcnn/{attempt}/images"), exist_ok=True)
    os.makedirs(os.path.join(images_path, f"faster_rcnn/{attempt}/plots"), exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Running on {device}, using {model_name}")
    print(f"This is {attempt}. attempt")

    model.load_state_dict(torch.load(os.path.join(models_path, model_name), map_location=device))
    model.to(device=device)

    split = "stage1_train"
    dataset = MyDataset(split=split, transforms=get_transforms(train=True, rescale_size=(256, 256)))
    trainset, evalset = random_split(dataset, [600, 70])

    train_loader = DataLoader(trainset, batch_size=1, num_workers=0, shuffle=True, collate_fn=my_collate)
    eval_loader = DataLoader(evalset, batch_size=1, num_workers=0, shuffle=False, collate_fn=my_collate)

    precision, recall, dice = evaluate(model, eval_loader, dist_threshold=3)

    print(f"Done, precision: {precision}, recall: {recall}, dice: {dice}")
    # dice_vec = []
    # prec_vec = []
    # rec_vec = []
    # # for thres in [0, 1, 1.5, 2, 2.5, 3, 3.5, 5]:
    # for thres in [50, 25, 10, 5, 2, 1.5, 1, 0.5, 0]:
    #     evaluate(thres)
    # print(rec_vec)
    # print(prec_vec)
    # rec_vec.reverse()
    #
    # fig, ax = plt.subplots()
    # ax.set_title('Multiple Samples with Different sizes')
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # # ax.boxplot(dice_vec)
    # ax.plot(rec_vec, prec_vec, 'r*')
    # plt.show()

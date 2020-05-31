import os
import time
import torch
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split

from data_utils import MyDataset, get_transforms
from models import Darknet
from utils import non_max_suppression

from conf.settings import BASE_DIR


models_path = os.path.join(BASE_DIR, "models")
images_path = os.path.join(BASE_DIR, "images")


def iou(box, clusters):
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def kmeans(boxes, k, dist=np.median):
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters

if __name__ == "__main__":
    attempt = 3

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}...")

    split = "stage1_train"
    dataset = MyDataset(split=split, transforms=get_transforms(train=True, rescale_size=(416, 416), yolo=True))
    boxes=[]
    for i, (image, targets) in enumerate(dataset):
        name = targets["name"]
        image = image
        for box in targets["boxes"]:
            boxes.append(box[3:5].tolist())
    boxes_array = np.asarray(boxes)
    clusters = kmeans(boxes_array, 9)
    roun_c = np.round(clusters * 416)
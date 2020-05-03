import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import random_split, DataLoader
from matplotlib import pyplot as plt
from yolo_v3.models import Darknet
from yolo_v3.utils.utils import weights_init_normal, non_max_suppression, xywh2xyxy

from utils.dataset import MyDataset, get_transform, my_collate

from conf.settings import BASE_DIR


def train():

    running_loss = 0

    model.train()
    for i, (image, targets) in enumerate(train_loader):
        name = targets[0]["name"]
        image = image[0].to(device=device)
        boxes = targets[0]["boxes"].to(device=device)

        loss, outputs = model(image, boxes)

        loss_item = loss.item()
        running_loss += loss_item

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch}, iteration: {i} of {len(trainset)}, loss: {loss_item}, image: {name}")

    training_loss.append(running_loss / len(trainset))


def evaluate():
    model.eval()
    for i, (image, targets) in enumerate(eval_loader):
        name = targets[0]["name"]
        image = image[0].to(device=device)

        outputs = model(image)
        outputs = non_max_suppression(outputs, conf_thres=0.5)

        boxes = outputs[0][:, 0:4]

        image_copy = Image.fromarray(image.cpu().numpy()[0, 0, :, :])
        if image_copy.mode != "RGB":
            image_copy = image_copy.convert("RGB")
        draw = ImageDraw.Draw(image_copy)
        for box in boxes:
            x0, y0, x1, y1 = box
            draw.rectangle([(x0, y0), (x1, y1)], outline=(255, 0, 255))
        image_copy.save(os.path.join(images_path, f"yolo_v3/{attempt}/images/{name}-{epoch}.png"))

        print(f"Epoch: {epoch}, iteration: {i} of {len(evalset)}, image: {name}")


def plot_losses():
    plt.figure(figsize=(16, 12), dpi=200)
    plt.plot(training_loss, 'r-', label="training_loss_sum",)
    plt.title("Training loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(images_path, f"yolo_v3/{attempt}/plots/training_loss_{attempt}.png"))
    plt.close()


if __name__ == "__main__":
    models_path = os.path.join(BASE_DIR, "models")
    images_path = os.path.join(BASE_DIR, "images")

    attempt = 1
    num_epoch = 10

    os.makedirs(models_path, exist_ok=True)
    os.makedirs(os.path.join(images_path, f"yolo_v3/{attempt}/images"), exist_ok=True)
    os.makedirs(os.path.join(images_path, f"yolo_v3/{attempt}/plots"), exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Running on {device}")
    print(f"This is {attempt}. attempt")

    model = Darknet("config/yolov3-custom.cfg").to(device)
    model.apply(weights_init_normal)
    # model.load_darknet_weights("weights/yolov3.weights")

    params = model.parameters()
    optimizer = torch.optim.Adam(params)

    split = "stage1_train"
    dataset = MyDataset(split=split, transforms=get_transform(train=True, rescale_size=(416, 416), yolo=True))
    trainset, evalset = random_split(dataset, [660, 10])  # this evalset is only for training progress demonstration

    train_loader = DataLoader(trainset, batch_size=1, num_workers=0, shuffle=True, collate_fn=my_collate)
    eval_loader = DataLoader(evalset, batch_size=1, num_workers=0, shuffle=False, collate_fn=my_collate)

    training_loss = []

    for epoch in range(num_epoch):
        train()
        evaluate()
        plot_losses()
        torch.save(model.state_dict(), os.path.join(models_path, f"yolo_v3_{attempt}.pt"))
    print("Done!")

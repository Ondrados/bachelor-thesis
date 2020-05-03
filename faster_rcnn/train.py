import os
import torch
from PIL import Image, ImageDraw
from torch.utils.data import random_split, DataLoader
from matplotlib import pyplot as plt
from faster_rcnn.models import model

from dataset import MyDataset, get_transform, my_collate

from conf.settings import BASE_DIR


def train():
    running_loss_sum = 0
    running_loss_cls1 = 0
    running_loss_reg1 = 0
    running_loss_cls2 = 0
    running_loss_reg2 = 0

    model.train()
    for i, (image, targets) in enumerate(train_loader):
        name = targets[0]["name"]
        image = image[0].to(device=device)
        targets = [{
            "boxes": targets[0]["boxes"].to(device=device),
            "labels": targets[0]["labels"].to(device=device),
            "name": name
        }]
        loss = model(image, targets)
        loss_sum = sum(lss for lss in loss.values())

        running_loss_sum += loss_sum
        running_loss_cls1 += loss["loss_objectness"]
        running_loss_cls2 += loss["loss_classifier"]
        running_loss_reg1 += loss["loss_rpn_box_reg"]
        running_loss_reg2 += loss["loss_box_reg"]

        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()

        print(f"Epoch: {epoch}, iteration: {i} of {len(trainset)}, loss: {loss_sum}, image: {name}")

    training_loss_sum.append(running_loss_sum / len(trainset))
    rpn_cls_loss.append(running_loss_cls1 / len(trainset))
    roi_cls_loss.append(running_loss_cls2 / len(trainset))
    rpn_reg_loss.append(running_loss_reg1 / len(trainset))
    roi_reg_loss.append(running_loss_reg2 / len(trainset))


def evaluate():
    model.eval()
    for i, (image, targets) in enumerate(eval_loader):
        name = targets[0]["name"]
        image = image[0].to(device=device)

        predictions = model(image)

        image_copy = Image.fromarray(image.cpu().numpy()[0, 0, :, :])
        if image_copy.mode != "RGB":
            image_copy = image_copy.convert("RGB")
        draw = ImageDraw.Draw(image_copy)
        for box, score in zip(predictions[0]["boxes"], predictions[0]["scores"]):
            x0, y0, x1, y1 = box
            draw.rectangle([(x0, y0), (x1, y1)], outline=(255, 0, 255))
        image_copy.save(os.path.join(images_path, f"faster_rcnn/{attempt}/images/{name}-{epoch}.png"))

        print(f"Epoch: {epoch}, iteration: {i} of {len(evalset)}, image: {name}")


def plot_losses():
    # TODO: show sum as average?
    # training_loss_sum_average = [x / 4 for x in training_loss_sum]
    plt.figure(figsize=(16, 12), dpi=200)
    plt.plot(training_loss_sum, 'r-', label="training_loss_sum",)
    plt.plot(rpn_cls_loss, label="rpn_cls_loss", )
    plt.plot(roi_cls_loss, label="roi_cls_loss", )
    plt.plot(rpn_reg_loss, label="rpn_reg_loss", )
    plt.plot(roi_reg_loss, label="roi_reg_loss", )
    plt.title("Training loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(images_path, f"faster_rcnn/{attempt}/plots/training_loss_{attempt}.png"))
    plt.close()


if __name__ == "__main__":
    models_path = os.path.join(BASE_DIR, "models")
    images_path = os.path.join(BASE_DIR, "images")

    attempt = 6
    num_epoch = 30

    os.makedirs(models_path, exist_ok=True)
    os.makedirs(os.path.join(images_path, f"faster_rcnn/{attempt}/images"), exist_ok=True)
    os.makedirs(os.path.join(images_path, f"faster_rcnn/{attempt}/plots"), exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Running on {device}")
    print(f"This is {attempt}. attempt")

    model.to(device=device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # optimizer = torch.optim.Adam(params, lr=0.0005, weight_decay=0)

    split = "stage1_train"
    dataset = MyDataset(split=split, transforms=get_transform(train=True, rescale_size=(256, 256)))
    trainset, evalset = random_split(dataset, [660, 10])  # this evalset is only for training progress demonstration

    train_loader = DataLoader(trainset, batch_size=1, num_workers=0, shuffle=True, collate_fn=my_collate)
    eval_loader = DataLoader(evalset, batch_size=1, num_workers=0, shuffle=False, collate_fn=my_collate)

    training_loss_sum = []
    rpn_cls_loss = []
    roi_cls_loss = []
    rpn_reg_loss = []
    roi_reg_loss = []

    for epoch in range(num_epoch):
        train()
        evaluate()
        plot_losses()
        torch.save(model.state_dict(), os.path.join(models_path, f"faster_rcnn_{attempt}.pt"))
    print("Done!")

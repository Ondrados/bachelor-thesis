import os
import torch
from PIL import Image, ImageDraw
from torch.utils.data import random_split
from matplotlib import pyplot as plt
from faster_rcnn.faster_rcnn import model

from dataset import MyDataset, get_transform

from settings import BASE_DIR

num_epoch = 30

models_path = os.path.join(BASE_DIR, "models")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}...")

model.to(device=device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.01,
                            momentum=0.9, weight_decay=0.0005)

dataset = MyDataset(split='stage1_train', transforms=get_transform(train=True))
trainset, valset = random_split(dataset, [500, 170])

training_loss = []


def train():
    i = 0
    running_loss = 0.0
    model.train()
    for image, targets in trainset:
        i += 1
        name = targets[0]["name"]
        print(f"Epoch: {epoch}, iteration: {i} of {len(trainset)}, image: {name} - train")
        image = image[None, :, :, :]
        image = image.to(device=device)
        targets = [{
                "boxes": targets[0]["boxes"].to(device=device),
                "labels": targets[0]["labels"].to(device=device)
                }]
        loss = model(image, targets)
        loss_sum = sum(lss for lss in loss.values())  #TODO: check this .items()
        print(loss_sum)
        running_loss += loss_sum

        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()

    loss = running_loss/len(trainset)
    training_loss.append(loss)


def evaluate():
    i = 0
    model.eval()
    for image, targets in valset:
        i += 1
        # every 10 images
        if (i % 10) == 0:
            name = targets[0]["name"]
            print(f"Epoch: {epoch}/{i} of {len(trainset)}, image {name} - eval")
            image = image[None, :, :, :]
            image = image.to(device=device)
            prediction = model(image)

            image2 = Image.fromarray(image.cpu().numpy()[0, 0, :, :])
            if image2.mode != "RGB":
                image2 = image2.convert("RGB")
            draw = ImageDraw.Draw(image2)
            for box, score in zip(prediction[0]["boxes"], prediction[0]["scores"]):
                x0, y0, x1, y1 = box
                draw.rectangle([(x0, y0), (x1, y1)], outline=(255, 0, 255))

            image2.save(f"faster_rcnn/images/{name}-{epoch}.png")


for epoch in range(num_epoch):
    train()
    evaluate()
    fig = plt.figure()
    plt.plot(training_loss, label="training_loss")
    plt.title("Training loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('faster_rcnn/plots/training_loss.png')
    torch.save(model.state_dict(), os.path.join(models_path, "faster_rcnn2.pt"))

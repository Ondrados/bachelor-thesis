import os
import torch
from torch.utils.data import random_split
from faster_rcnn.faster_rcnn import model

from dataset import MyDataset, get_transform

from settings import BASE_DIR

num_epoch = 30

models_path = os.path.join(BASE_DIR, "models")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}...")

model.to(device=device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.01,
                            momentum=0.9, weight_decay=0.0005)

dataset = MyDataset(split='stage1_train', transforms=get_transform(train=True))
trainset, valset = random_split(dataset, [500, 170])


def train():
    i = 0
    model.train()
    for image, targets in trainset:
        i += 1
        name = targets[0]["name"]
        print(f"Epoch: {epoch}, iteration: {i} of {len(trainset)}, image: {name} - train")
        image = image[None, :, :, :]
        image.to(device=device)

        loss = model(image, targets)
        loss_sum = sum(lss for lss in loss.values())
        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()

        # TODO: add loss save


def evaluate():
    i = 0
    model.eval()
    for image, targets in valset:
        i += 1
        name = targets[0]["name"]
        print(f"Epoch: {epoch}/{i} of {len(trainset)}, image {name} - eval")
        image.to(device=device)
        image = image[None, :, :, :]
        predictions = model(image)

        # TODO: add image save


for epoch in range(num_epoch):
    train()
    evaluate()
    torch.save(model.state_dict(), os.path.join(models_path, "faster_rcnn1.pt"))

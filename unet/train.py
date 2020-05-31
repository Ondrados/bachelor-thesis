import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from data_utils import MyDataset, get_transforms, my_collate
from models import UNet

from conf.settings import BASE_DIR


def train():
    running_loss = 0.0
    model.train()
    for i, (input, mask, name) in enumerate(train_loader):
        input, mask = input[0].to(device=device)[None,:,:,:], mask[0].to(device=device)[None,:,:,:]

        input.requires_grad = True
        mask.requires_grad = True
        optimizer.zero_grad()

        output = model(input)
        loss = criterion(output, mask)
        loss_item = loss.item()

        running_loss += loss_item

        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch}, iteration: {i} of {len(train_loader)}, loss: {loss_item}, image: {name[0]}")

    training_loss.append(running_loss / len(train_loader))

    # fig = plt.figure()
    # fig.add_subplot(1, 3, 1)
    # plt.imshow(inputs[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
    # fig.add_subplot(1, 3, 2)
    # plt.imshow(masks[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
    # fig.add_subplot(1, 3, 3)
    # plt.imshow(outputs[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
    # #plt.show(block=True)
    # plt.savefig('images/train_output.png')


def evaluate():
    running_loss_eval = 0.0
    model.eval()
    for i, (input, mask, name) in enumerate(eval_loader):
        input, mask = input[0].to(device=device)[None,:,:,:], mask[0].to(device=device)[None,:,:,:]

        with torch.no_grad():
            output = model(input)
            # torch.where(outputs > 0.5, torch.ones(1).cuda(),torch.zeros(1).cuda())
            loss = criterion(output, mask)
            loss_item = loss.item()

        running_loss_eval += loss_item

        print(f"Eval: {epoch}, iteration: {i} of {len(eval_loader)}, loss: {loss_item}, image: {name[0]}")

    eval_loss.append(running_loss_eval / len(eval_loader))
    # fig = plt.figure()
    # fig.add_subplot(1, 3, 1)
    # plt.imshow(inputs[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
    # fig.add_subplot(1, 3, 2)
    # plt.imshow(masks[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
    # fig.add_subplot(1, 3, 3)
    # plt.imshow(outputs[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
    # # plt.show(block=True)
    # plt.savefig('images/eval_output.png')


def plot_losses():
    plt.figure(figsize=(12, 8), dpi=200)
    plt.plot(training_loss, 'r-', label="training_loss",)
    plt.plot(eval_loss, 'b-', label="validation_loss", )
    plt.title("Training and validation loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(images_path, f"unet/{attempt}/plots/training_eval_loss_{attempt}.png"))
    plt.close()
    plt.figure(figsize=(12, 8), dpi=200)
    plt.plot(training_loss, 'r-', label="training_loss", )
    plt.title("Training loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(images_path, f"unet/{attempt}/plots/training_loss_{attempt}.png"))
    plt.close()


if __name__ == "__main__":
    models_path = os.path.join(BASE_DIR, "models")
    images_path = os.path.join(BASE_DIR, "images")

    attempt = 2
    num_epoch = 50

    os.makedirs(models_path, exist_ok=True)
    os.makedirs(os.path.join(images_path, f"unet/{attempt}/images"), exist_ok=True)
    os.makedirs(os.path.join(images_path, f"unet/{attempt}/plots"), exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Running on {device}")
    print(f"This is {attempt}. attempt")

    batch = 1
    split = "stage1_train"
    dataset = MyDataset(split=split, model="unet")
    # dataset = MyDataset(split="stage1_train")
    trainset, evalset = random_split(dataset, [600, 70])

    train_loader = DataLoader(trainset, batch_size=batch, num_workers=1, shuffle=True, drop_last=True)
    eval_loader = DataLoader(evalset, batch_size=batch, num_workers=1, shuffle=True, drop_last=True)

    model = UNet(n_channels=1, n_classes=1)
    model.to(device=device)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    # criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    training_loss = []
    eval_loss = []

    for epoch in range(num_epoch):
        # train()
        # evaluate()
        plot_losses()
        if (epoch % 10) == 0:
            torch.save(model.state_dict(), os.path.join(models_path, f"unet_{attempt}_{epoch}.pt"))
        else:
            torch.save(model.state_dict(), os.path.join(models_path, f"unet_{attempt}.pt"))
    print("Done!")

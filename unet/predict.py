
# pick one image from the test set
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from utils.dataset import MyDataset
from unet import UNet


batch = 8
dataset = MyDataset(split="stage1_train")
trainset, valset = random_split(dataset, [500, 170])

train_loader = DataLoader(
    trainset, batch_size=batch, num_workers=1, shuffle=True, drop_last=True)
val_loader = DataLoader(
    valset, batch_size=batch, num_workers=1, shuffle=True, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

net = UNet(n_channels=1, n_classes=1)
net.to(device=device)

criterion = nn.MSELoss()
# criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

validation_loss = []

def evaluate():
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            # get the inputs and masks
            inputs, masks = data[0].to(device=device), data[1].to(device=device)

            net.eval()

            # only forward
            outputs = net(inputs)
            loss = criterion(outputs, masks)

            # print statistics
            print(f'{i}/{len(val_loader)}/{epoch}/eval')
            running_loss += loss.item()
        running_loss = running_loss / len(val_loader)
        validation_loss.append(running_loss)

        fig = plt.figure()
        fig.add_subplot(1, 3, 1)
        plt.imshow(inputs[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
        fig.add_subplot(1, 3, 2)
        plt.imshow(masks[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
        fig.add_subplot(1, 3, 3)
        plt.imshow(outputs[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
        # plt.show(block=True)
        plt.savefig('images/eval_output.png')


for epoch in range(50):  # loop over the dataset multiple times
    evaluate()
    fig = plt.figure()
    plt.plot(training_loss, label="training_loss")
    plt.plot(validation_loss, label="validation_loss")
    plt.legend()
    # plt.show()
    plt.savefig('images/training_loss.png')
    plt.close("all")
    torch.save(net.state_dict(), '/home/ubmi/Documents/cnn-cells/cnn-cells/models/my_model.pt')

print('Training finished!!!')
img, _ = dataset_test[0]
# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
from torch.utils import data

from dataset import MyDataset
from unet import UNet

batch = 8
dataset = MyDataset(split="stage1_train")
trainloader = data.DataLoader(
    dataset, batch_size=batch, num_workers=1, shuffle=True, drop_last=True,)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

net = UNet(n_channels=1, n_classes=1)
net.to(device=device)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

training_loss = []
for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, masks = data[0].to(device=device), data[1].to(device=device)

        inputs.requires_grad = True
        masks.requires_grad = True

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        # print statistics
        print(f'{i}/{len(trainloader)}/{epoch}')
        running_loss += loss.item()
        if i % 100 == 0:    # every 1OO it
            print('[%d, %d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / ((i+1)*100)))
    running_loss = running_loss / len(trainloader)
    training_loss.append(running_loss)

    plt.plot(training_loss)
    #plt.show()
    plt.savefig('training_loss.png')

    fig = plt.figure(figsize=(1, 3))
    fig.add_subplot(1, 3, 1)
    plt.imshow(inputs[0, 0, :, :].detach().cpu().numpy(), cmap="gray", aspect='auto')
    fig.add_subplot(1, 3, 2)
    plt.imshow(masks[0, 0, :, :].detach().cpu().numpy(), cmap="gray", aspect='auto')
    fig.add_subplot(1, 3, 3)
    plt.imshow(outputs[0, 0, :, :].detach().cpu().numpy(), cmap="gray", aspect='auto')
    #plt.show(block=True)
    plt.savefig('output.png')

    torch.save(net.state_dict(), '/home/ubmi/Documents/cnn-cells/cnn-cells/my_model.pt')

print('Training finished!!!')

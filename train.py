import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

from dataset import MyDataset
from unet import UNet

batch = 1
dataset = MyDataset(split="stage1_train")
trainloader = data.DataLoader(
    dataset, batch_size=batch, num_workers=0, shuffle=True, drop_last=True,)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

net = UNet(n_channels=1, n_classes=1)
net.to(device=device)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, masks = data[0].to(device=device), data[1].to(device=device)
        inputs = inputs.unsqueeze(1)
        masks = masks.unsqueeze(1)

        inputs.requires_grad = True
        masks.requires_grad = True

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, masks)
        print(loss.item())
        loss.backward()
        optimizer.step()

        print(f'{i}/{len(trainloader)}/{epoch}')
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import MyDataset
from unet import UNet
from skimage.feature import peak_local_max
from skimage.morphology import extrema

from skimage.measure import label
from skimage import color

dataset = MyDataset(split="stage1_train",path_to_data = '/Users/ondra/Dev/Personal/cnn-cells/data-science-bowl-2018')

data_loader = DataLoader(
    dataset, batch_size=1, num_workers=1, shuffle=True, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = UNet(n_channels=1, n_classes=1)
net.load_state_dict(torch.load('/Users/ondra/Dev/Personal/cnn-cells/models/my_model2.pt',map_location=device))
net.to(device)

with torch.no_grad():
    # get the inputs and masks
    data = next(iter(data_loader))
    inputs, masks = data[0].to(device=device), data[1].to(device=device)

    net.eval()

    outputs = net(inputs)
out = outputs[0, 0, :, :].numpy()
mask = masks[0, 0, :, :].numpy()
array = np.where(out > 0.3, 1, 0)
diff = mask-array



h = 0.3
h_maxima = extrema.h_maxima(out, h)
label_h_maxima = label(h_maxima)
overlay_h = color.label2rgb(label_h_maxima, inputs[0, 0, :, :], alpha=0.7, bg_label=0,
                            bg_color=None, colors=[(1, 0, 0)])

coordinates = peak_local_max(label_h_maxima, indices=True)

mask_h = 0.3
mask_h_maxima = extrema.h_maxima(out, h)
mask_label_h_maxima = label(h_maxima)
overlay_h = color.label2rgb(label_h_maxima, inputs[0, 0, :, :], alpha=0.7, bg_label=0,
                            bg_color=None, colors=[(1, 0, 0)])

coordinates = peak_local_max(label_h_maxima, indices=True)
# fig = plt.figure()
# fig.add_subplot(1, 2, 1)
# plt.title("Input image")
# plt.imshow(inputs[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
# fig.add_subplot(1, 2, 2)
# plt.title("Output image")
# plt.imshow(outputs[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
# plt.savefig('images/out.png')
# # plt.show(block=True)
# plt.close()
#
# fig = plt.figure()
# fig.add_subplot(1, 2, 1)
# plt.title("Output image")
# plt.imshow(outputs[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
# fig.add_subplot(1, 2, 2)
# plt.title("Thresholding")
# plt.imshow(array, cmap="gray")
# plt.savefig('images/tres.png')
# # plt.show(block=True)
# plt.close()
#
# fig = plt.figure()
# fig.add_subplot(1, 2, 1)
# plt.title("Input image")
# plt.imshow(inputs[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
# fig.add_subplot(1, 2, 2)
# plt.title("Detection")
# plt.imshow(inputs[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
# plt.plot(coordinates[:,1],coordinates[:,0], 'r+')

plt.imshow(inputs[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
plt.plot(coordinates[:, 1], coordinates[:, 0], 'r+', markersize=10)
# plt.savefig('images/detect6.png')
plt.show(block=True)
plt.close()

# fig.add_subplot(1, 4, 4)
# plt.title("Detection")
# plt.imshow(inputs[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
# plt.plot(coordinates[:,1],coordinates[:,0], 'r+')

# plt.imshow(inputs[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
# plt.plot(coordinates[:,1],coordinates[:,0], 'r+', markersize=10)
# plt.show(block=True)
# plt.savefig('images/train_output.png')
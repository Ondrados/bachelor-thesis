import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_utils import MyDataset, MyTestDataset
from models import UNet
from skimage.feature import peak_local_max
from skimage.morphology import extrema

from skimage.measure import label
from skimage import color

from conf.settings import BASE_DIR

images_path = os.path.join(BASE_DIR, "images")

# NOTE: probably delete?

if __name__ == "__main__":
    # torch.manual_seed(61)
    dataset = MyTestDataset(model="unet")

    data_loader = DataLoader(
        dataset, batch_size=1, num_workers=0, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = UNet(n_channels=1, n_classes=1)
    # net.load_state_dict(torch.load('/Users/ondra/Dev/Personal/cnn-cells/models/my_model2.pt',map_location=device))
    net.load_state_dict(torch.load('/Users/ondra/Dev/Personal/cnn-cells/models/unet_2_15.pt', map_location=device))
    net.to(device)

    for i, (inputs, name) in enumerate(data_loader):
        if name[0] != "bdc789019cee8ddfae20d5f769299993b4b330b2d38d1218646cf89e77fbbd4d":
            continue

        with torch.no_grad():
            # get the inputs and masks
            inputs = inputs[0].to(device=device)[None,:,:,:]

            net.eval()

            outputs = net(inputs)
        out = outputs[0, 0, :, :].numpy()
        # mask = masks[0, 0, :, :].numpy()
        array = np.where(out < 0.3, 0, out)
        array = out
        # diff = mask-array



        h = 0.2
        h_maxima = extrema.h_maxima(array, h)
        label_h_maxima = label(h_maxima)
        overlay_h = color.label2rgb(label_h_maxima, inputs[0, 0, :, :], alpha=0.7, bg_label=0,
                                    bg_color=None, colors=[(1, 0, 0)])

        coordinates = peak_local_max(array, indices=True)
        coordinates_h = peak_local_max(h_maxima, indices=True)


        fig = plt.figure(dpi=300)
        # ax1 = fig.add_subplot(2, 3, 1)
        # ax1.imshow(inputs[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
        # # ax1.plot(coordinates[:, 1], coordinates[:, 0], 'r+', markersize=10)
        # ax2 = fig.add_subplot(2, 3, 2)
        # ax2.imshow(array, cmap="jet")
        # ax3 = fig.add_subplot(2, 3, 3)
        # # ax3.imshow(inputs[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
        # # ax3.imshow(outputs[0, 0, :, :].detach().cpu().numpy(), cmap="jet", alpha=0.3)
        # ax3.imshow(inputs[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
        # ax3.plot(coordinates[:, 1], coordinates[:, 0], 'r+', markersize=10)

        # ax4 = fig.add_subplot(1, 1, 1)
        # ax4.imshow(array, cmap="jet")
        # # ax4.plot(coordinates[:, 1], coordinates[:, 0], 'r+', markersize=10)
        # ax5 = fig.add_subplot(1, 1, 1)
        # ax5.imshow(h_maxima, cmap="gray")
        ax6 = fig.add_subplot(1, 1, 1)
        ax6.imshow(inputs[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
        ax6.plot(coordinates_h[:, 1], coordinates_h[:, 0], 'r+', markersize=10)
        plt.show()


        # print(len(coordinates), len(coordinates_h))

        # mask_h = 0.3
        # mask_h_maxima = extrema.h_maxima(out, h)
        # mask_label_h_maxima = label(h_maxima)
        # overlay_h = color.label2rgb(label_h_maxima, inputs[0, 0, :, :], alpha=0.7, bg_label=0,
        #                             bg_color=None, colors=[(1, 0, 0)])
        #
        # coordinates = peak_local_max(mask_label_h_maxima, indices=True)
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

        # plt.imshow(inputs[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
        # plt.plot(coordinates[:, 1], coordinates[:, 0], 'r+', markersize=10)
        # # plt.savefig('images/detect6.png')
        # plt.show(block=True)
        # plt.close()

        # plt.savefig(os.path.join(images_path, f"unet/2/images/unet_hmax3.png"))

        # fig.add_subplot(1, 4, 4)
        # plt.title("Detection")
        # plt.imshow(inputs[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
        # plt.plot(coordinates[:,1],coordinates[:,0], 'r+')

        # plt.imshow(inputs[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
        # plt.plot(coordinates[:,1],coordinates[:,0], 'r+', markersize=10)
        # plt.show(block=True)
        # plt.savefig('images/train_output.png')
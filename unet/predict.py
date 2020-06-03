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

models_path = os.path.join(BASE_DIR, "models")
images_path = os.path.join(BASE_DIR, "images")


if __name__ == "__main__":
    # torch.manual_seed(61)
    unet_name = "unet_2_15.pt"

    dataset = MyTestDataset(model="unet")

    data_loader = DataLoader(
        dataset, batch_size=1, num_workers=0, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(n_channels=1, n_classes=1)
    model.load_state_dict(torch.load(os.path.join(models_path, unet_name), map_location=device))
    model.to(device=device)

    for i, (inputs, name) in enumerate(data_loader):
        # if name[0] != "bdc789019cee8ddfae20d5f769299993b4b330b2d38d1218646cf89e77fbbd4d":
        #     continue
        with torch.no_grad():
            inputs = inputs[0].to(device=device)[None,:,:,:]
            model.eval()
            outputs = model(inputs)
        out = outputs[0, 0, :, :].numpy()

        h = 0.3
        h_maxima = extrema.h_maxima(out, h)
        label_h_maxima = label(h_maxima)
        coordinates_h = peak_local_max(h_maxima, indices=True)

        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(inputs[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
        ax.plot(coordinates_h[:, 1], coordinates_h[:, 0], 'r+', markersize=10)
        plt.show()
        break

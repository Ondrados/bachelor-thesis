import os
import time
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


def predict(model, dataloader):
    model.eval()
    for i, (inputs, name) in enumerate(dataloader):
        name = name[0]
        start_time = time.time()
        with torch.no_grad():
            inputs = inputs[0].to(device=device)[None,:,:,:]
            outputs = model(inputs)
        out = outputs[0, 0, :, :].numpy()

        h = 0.3
        h_maxima = extrema.h_maxima(out, h)
        coordinates_h = peak_local_max(h_maxima, indices=True)
        elapsed_time = time.time() - start_time

        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(inputs[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
        ax.plot(coordinates_h[:, 1], coordinates_h[:, 0], 'r+', markersize=10)
        print(f"{name}, time: {elapsed_time}")
        plt.show()
        break



if __name__ == "__main__":
    # torch.manual_seed(61)
    unet_name = "unet_2_15.pt"

    dataset = MyTestDataset(model="unet")

    test_loader = DataLoader(
        dataset, batch_size=1, num_workers=0, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(n_channels=1, n_classes=1)
    model.load_state_dict(torch.load(os.path.join(models_path, unet_name), map_location=device))
    model.to(device=device)

    predict(model, test_loader)

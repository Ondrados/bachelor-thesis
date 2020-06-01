import os
import time
import torch
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split

from data_utils import MyTestDataset, get_test_transforms
from models import Darknet
from utils import non_max_suppression

from conf.settings import BASE_DIR


models_path = os.path.join(BASE_DIR, "models")
images_path = os.path.join(BASE_DIR, "images")

if __name__ == "__main__":
    attempt = 4

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}...")

    model = Darknet(os.path.join(BASE_DIR, "yolo_v3/config/yolov3-custom.cfg")).to(device)
    model.load_state_dict(torch.load(os.path.join(models_path, "yolo_v3_4_20.pt"), map_location=device))

    dataset = MyTestDataset(split='stage1_test', transforms=get_test_transforms(rescale_size=(416, 416)))

    test_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

    model.eval()
    for i, (image, targets) in enumerate(test_loader):
        image = image[0].to(device=device)
        name = targets["name"][0]
        start_time = time.time()
        with torch.no_grad():
            outputs = model(image)
        outputs = non_max_suppression(outputs, conf_thres=0.5)
        elapsed_time = time.time() - start_time
        if outputs[0] is not None:
            boxes = outputs[0][:, 0:4]
        else:
            continue

        image_copy = Image.fromarray(image.cpu().numpy()[0, 0, :, :])
        if image_copy.mode != "RGB":
            image_copy = image_copy.convert("RGB")
        draw = ImageDraw.Draw(image_copy)
        for box in boxes:
            x0, y0, x1, y1 = box
            draw.rectangle([(x0, y0), (x1, y1)], outline=(255, 0, 255))
        # image_copy.show()
        # image_copy.save(os.path.join(images_path, f"yolo_v3/{attempt}/images/{name}.png"))
        print(f"{name}, time: {elapsed_time}")
        plt.imshow(image_copy)
        plt.show()


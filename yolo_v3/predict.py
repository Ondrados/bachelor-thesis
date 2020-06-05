import os
import time
import torch
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split

from data_utils import MyTestDataset, get_test_transforms
from yolo_v3.models import Darknet
from utils import non_max_suppression, resize_boxes

from conf.settings import BASE_DIR


models_path = os.path.join(BASE_DIR, "models")
images_path = os.path.join(BASE_DIR, "images")


def predict(model, dataloader=None, image=None):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if image is not None:
        image = image[0].to(device=device)
        start_time = time.time()
        with torch.no_grad():
            outputs = model(image)
        outputs = non_max_suppression(outputs, conf_thres=0.5, nms_thres=0.2)
        elapsed_time = time.time() - start_time
        if outputs[0] is not None:
            boxes = outputs[0][:, 0:4]
            boxes = resize_boxes(boxes, (416, 416), (256, 256))
        pred_x = []
        pred_y = []
        for box in boxes:
            x0, y0, x1, y1 = box
            x = ((x0 + x1) / 2).tolist()
            y = ((y0 + y1) / 2).tolist()
            pred_x.append(x)
            pred_y.append(y)
        image = Image.fromarray(image.cpu().numpy()[0, 0, :, :]).convert("RGB").resize((256, 256))
        return image, pred_x, pred_y

    for i, (image, targets) in enumerate(dataloader):
        image = image[0].to(device=device)
        name = targets["name"][0]
        start_time = time.time()
        with torch.no_grad():
            outputs = model(image)
        outputs = non_max_suppression(outputs, conf_thres=0.5)
        elapsed_time = time.time() - start_time
        if outputs[0] is not None:
            boxes = outputs[0][:, 0:4]
            boxes = resize_boxes(boxes, (416, 416), (256, 256))
        else:
            continue
        image_copy = Image.fromarray(image.cpu().numpy()[0, 0, :, :]).resize((256, 256))
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
        break


if __name__ == "__main__":
    # torch.manual_seed(1)
    attempt = 4
    model_name = "yolo_v3_4_20.pt"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}...")

    model = Darknet(os.path.join(BASE_DIR, "yolo_v3/config/yolov3-custom.cfg")).to(device)
    model.load_state_dict(torch.load(os.path.join(models_path, model_name), map_location=device))

    dataset = MyTestDataset(split='stage1_test', transforms=get_test_transforms(rescale_size=(416, 416)))
    test_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)

    predict(model, test_loader)



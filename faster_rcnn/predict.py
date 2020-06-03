import os
import time
import torch
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from torch.utils.data import random_split, DataLoader

from data_utils import MyTestDataset, get_test_transforms

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
            predictions = model(image)
        elapsed_time = time.time() - start_time
        pred_x = []
        pred_y = []
        for box, score in zip(predictions[0]["boxes"], predictions[0]["scores"]):
            if score > 0.5:
                x0, y0, x1, y1 = box
                x = ((x0 + x1) / 2).tolist()
                y = ((y0 + y1) / 2).tolist()
                pred_x.append(x)
                pred_y.append(y)
        image = Image.fromarray(image.cpu().numpy()[0, 0, :, :]).convert("RGB")
        return image, pred_x, pred_y

    for i, (image, targets) in enumerate(dataloader):
        image = image[0].to(device=device)
        name = targets["name"][0]
        start_time = time.time()
        with torch.no_grad():
            predictions = model(image)
        elapsed_time = time.time() - start_time
        image_copy = Image.fromarray(image.cpu().numpy()[0, 0, :, :])
        if image_copy.mode != "RGB":
            image_copy = image_copy.convert("RGB")
        draw = ImageDraw.Draw(image_copy)
        for box, score in zip(predictions[0]["boxes"], predictions[0]["scores"]):
            if score > 0.5:
                x0, y0, x1, y1 = box
                draw.rectangle([(x0, y0), (x1, y1)], outline=(255, 0, 255))
        # image_copy.show()
        # image_copy.save(os.path.join(images_path, f"faster_rcnn/{attempt}/images/{name}.png"))
        print(f"{name}, time: {elapsed_time}")
        plt.imshow(image_copy)
        plt.show()
        break


if __name__ == "__main__":
    # torch.manual_seed(4)
    from models import model

    attempt = 7
    model_name = "faster_rcnn_7_30.pt"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}...")

    model.load_state_dict(torch.load(os.path.join(models_path, "faster_rcnn_7_30.pt"), map_location=device))
    dataset = MyTestDataset(split='stage1_test', transforms=get_test_transforms(rescale_size=(256, 256)))
    test_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)

    predict(model, test_loader)

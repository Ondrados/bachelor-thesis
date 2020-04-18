import os
import torch
from PIL import Image, ImageDraw
from torch.utils.data import random_split
from matplotlib import pyplot as plt
from faster_rcnn.faster_rcnn import model

from dataset import MyDataset, get_transform

from settings import BASE_DIR

num_epoch = 30

models_path = os.path.join(BASE_DIR, "models")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}...")

model.load_state_dict(torch.load(os.path.join(models_path, "faster_rcnn1.pt"), map_location=device))
dataset = MyDataset(split='stage1_train', transforms=get_transform(train=True))
trainset, valset = random_split(dataset, [500, 170])


def evaluate():
    i = 0
    for image, targets in valset:
        i += 1
        model.eval()
        # every 10 images
        if (i % 10) == 0:
            name = targets[0]["name"]
            print(f"{i} of {len(valset)}, image {name} - eval")
            image = image[None, :, :, :]
            image = image.to(device=device)

            prediction = model(image)

            image2 = Image.fromarray(image.cpu().numpy()[0, 0, :, :])
            if image2.mode != "RGB":
                image2 = image2.convert("RGB")
            draw = ImageDraw.Draw(image2)
            for box, score in zip(prediction[0]["boxes"], prediction[0]["scores"]):
                x0, y0, x1, y1 = box
                draw.rectangle([(x0, y0), (x1, y1)], outline=(255, 0, 255))
            #image2.show()
            image2.save(f"faster_rcnn/images/{name}.png")
            break


if __name__ == "__main__":
    evaluate()

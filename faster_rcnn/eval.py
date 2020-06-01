import os
import torch
from PIL import Image, ImageDraw
from torch.utils.data import random_split, DataLoader
from matplotlib import pyplot as plt
from models import model

from data_utils import MyDataset, get_transforms, my_collate

from conf.settings import BASE_DIR

models_path = os.path.join(BASE_DIR, "models")
images_path = os.path.join(BASE_DIR, "images")


def evaluate():
    model.eval()
    for i, (image, targets) in enumerate(eval_loader):

        name = targets[0]["name"]
        image = image[0].to(device=device)
        targets = [{
            "boxes": targets[0]["boxes"].to(device=device),
            "labels": targets[0]["labels"].to(device=device),
            "name": name
        }]

        with torch.no_grad():
            predictions = model(image)

        image_copy = Image.fromarray(image.cpu().numpy()[0, 0, :, :])
        if image_copy.mode != "RGB":
            image_copy = image_copy.convert("RGB")
        draw = ImageDraw.Draw(image_copy)
        for box in targets[0]["boxes"]:
            x0, y0, x1, y1 = box
            draw.rectangle([(x0, y0), (x1, y1)], outline=(0, 255, 0))
        for box, score in zip(predictions[0]["boxes"], predictions[0]["scores"]):
            if score > 0.5:
                x0, y0, x1, y1 = box
                draw.rectangle([(x0, y0), (x1, y1)], outline=(255, 0, 255))
        # image_copy.show()
        # image_copy.save(os.path.join(images_path, f"faster_rcnn/{attempt}/images/{name}.png"))
        plt.imshow(image_copy)
        plt.show()

        print(f"Iteration: {i} of {len(eval_loader)}, image: {name}")


if __name__ == "__main__":
    from models import model

    attempt = 7
    model_name = "faster_rcnn_7_30.pt"

    os.makedirs(models_path, exist_ok=True)
    os.makedirs(os.path.join(images_path, f"faster_rcnn/{attempt}/images"), exist_ok=True)
    os.makedirs(os.path.join(images_path, f"faster_rcnn/{attempt}/plots"), exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Running on {device}, using {model_name}")
    print(f"This is {attempt}. attempt")

    model.load_state_dict(torch.load(os.path.join(models_path, model_name), map_location=device))
    model.to(device=device)

    split = "stage1_train"
    dataset = MyDataset(split=split, transforms=get_transforms(train=True, rescale_size=(256, 256)))
    trainset, evalset = random_split(dataset, [600, 70])

    train_loader = DataLoader(trainset, batch_size=1, num_workers=0, shuffle=True, collate_fn=my_collate)
    eval_loader = DataLoader(evalset, batch_size=1, num_workers=0, shuffle=False, collate_fn=my_collate)

    evaluate()

import os
import torch
from PIL import Image, ImageDraw
from torch.utils.data import random_split, DataLoader

from data_utils import MyTestDataset, get_test_transforms

from conf.settings import BASE_DIR


models_path = os.path.join(BASE_DIR, "models")
images_path = os.path.join(BASE_DIR, "images")

if __name__ == "__main__":
    from models import model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}...")

    model.load_state_dict(torch.load(os.path.join(models_path, "faster_rcnn_6.pt"), map_location=device))

    dataset = MyTestDataset(split='stage1_test', transforms=get_test_transforms(rescale_size=(256, 256)))

    test_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)

    model.eval()
    for i, (image, targets) in enumerate(test_loader):
        image = image[0].to(device=device)
        name = targets["name"]

        predictions = model(image)
        # TODO: add non_max_supression

        image_copy = Image.fromarray(image.cpu().numpy()[0, 0, :, :])
        if image_copy.mode != "RGB":
            image_copy = image_copy.convert("RGB")
        draw = ImageDraw.Draw(image_copy)
        for box, score in zip(predictions[0]["boxes"], predictions[0]["scores"]):
            x0, y0, x1, y1 = box
            draw.rectangle([(x0, y0), (x1, y1)], outline=(255, 0, 255))
        # image_copy.save(os.path.join(images_path, f"faster_rcnn/{attempt}/images/{name}-{epoch}.png"))
        image_copy.show()
        if i == 5:
            break

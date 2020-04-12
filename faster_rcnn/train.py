import torch
from faster_rcnn import model

from dataset import MyDataset

models_path = os.path.join(BASE_DIR, "models")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}..")

model.to(device=device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.01,
                            momentum=0.9, weight_decay=0.0005)

dataset = MyDataset(split='stage1_train', transforms=get_transform(train=True))
trainset, valset = random_split(dataset, [500, 170])


def train():
    model.train()
    for image, targets in trainset:
        image.to(device=device)
        targets.to(device=device)

        loss = model(image, targets)
        loss_sum = sum(lss for lss in loss.values())
        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()


def evaluate():
    model.eval()
    for image, targets in valset:
        image.to(device=device)
        targets.to(device=device)
        predictions = model(image)


for epoch in range(50):
    train()
    evaluate()
    torch.save(model.state_dict(), os.path.join(models_path, "faster_rcnn1.pt"))

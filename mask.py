import matplotlib.pyplot as plt
from dataset import MyDataset

image, mask = MyDataset(path_to_data = '/Users/ondra/Dev/Personal/cnn-cells/data-science-bowl-2018').__getitem__(1)
import os
import glob
from skimage.io import imread
from matplotlib import pyplot as plt
import numpy as np

path_to_data='/Users/ondra/Dev/Personal/cnn-cells/data-science-bowl-2018'
split='stage1_train'
path = path_to_data + '/' + split

id_list = os.listdir(path)
image_list = []
mask_list = []

for id in id_list:
    images = glob.glob(path + '/' + id + '/images/*png')
    masks = glob.glob(path + '/' + id + '/masks/*png')
    image_list.extend(images)
    mask_list.append(masks)

image_path = image_list[0]
masks_path = mask_list[0]
print(image_path)
image = imread(image_path)
# comb_mask = np.zeros((256,256), dtype=float)

comb_mask = None
for path in masks_path:
    mask = imread(path)
    if comb_mask is None:
        comb_mask = np.zeros_like(mask, dtype=np.float32())
    comb_mask += mask

fig = plt.figure(figsize=(1, 2))
fig.add_subplot(1, 2, 1)
plt.imshow(image)
fig.add_subplot(1, 2, 2)
plt.imshow(comb_mask, cmap='binary')

plt.show(block=True)

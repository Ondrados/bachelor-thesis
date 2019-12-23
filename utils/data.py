import os
import glob
from PIL import Image
from skimage.io import imread
from matplotlib import pyplot as plt
import numpy as np

path_to_data = '/Users/ondra/Dev/Personal/cnn-cells/data-science-bowl-2018'
split = 'stage1_train'
path = path_to_data + '/' + split

path_id_list = glob.glob(os.path.join(path, '*'))

image_list = []
mask_list = []

for path in path_id_list:
    images = glob.glob(path + '/images/*png')
    masks = glob.glob(path + '/masks/*png')
    image_list.extend(images)
    mask_list.append(masks)

# with open('id_list.txt', 'w') as f:
#     for item in id_list:
#         f.write("%s\n" % item)
# with open('image_list.txt', 'w') as f:
#     for item in image_list:
#         f.write("%s\n" % item)
# with open('mask_list.txt', 'w') as f:
#     for item in mask_list:
#         f.write("%s\n" % item)

image_path = image_list[464]
masks_path = mask_list[464]
image = Image.open(image_path)

comb_mask = None
for path in masks_path:
    mask = Image.open(path)
    if comb_mask is None:
        comb_mask = np.zeros_like(mask, dtype=np.float32())
    comb_mask += mask

Image.fromarray(comb_mask)

fig = plt.figure(figsize=(1, 2))
fig.add_subplot(1, 2, 1)
plt.imshow(image)
fig.add_subplot(1, 2, 2)
plt.imshow(comb_mask, cmap='binary')

plt.show(block=True)

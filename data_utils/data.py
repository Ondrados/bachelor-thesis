import os
import glob
from PIL import Image
from skimage.io import imread
from skimage.measure import regionprops
from matplotlib import pyplot as plt
import numpy as np

fig = plt.figure(dpi=300)
ax1 = fig.add_subplot(1, 1, 1)
ax1.imshow(image_copy)
ax2 = fig.add_subplot(2, 2, 2)
ax2.imshow(image_copy2)
ax3 = fig.add_subplot(2, 2, 3)
ax3.imshow(image_copy3)
plt.show()

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

fig, ax = plt.subplots()
num_bins = 50

# the histogram of the data
n, bins, patches = ax.hist(dists, num_bins, density=1)
ax.vlines(5, 0, 0.35, 'r', linestyles='dashed')
plt.text(5, -0.02, "5", color='red', fontsize=10)
ax.set_xlabel('Vzálednost od pravdivé polohy')
ax.set_ylabel('Hustota pravděpodobnosti')
ax.set_title('Rozložení vzdáleností mezi detekovanou a pravdivou polohou')
fig.tight_layout()
plt.show()

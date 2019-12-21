import os
import glob

path_to_data='/Users/ondra/Dev/Personal/cnn-cells/data-science-bowl-2018'
split='stage1_train'
path = path_to_data + '/' + split

id_list = os.listdir(path)
image_list = []
mask_list = []

for id in self.id_list:
    images = glob.glob(self.path + '/' + id + '/images/*png')
    masks = glob.glob(self.path + '/' + id + '/masks/*png')
    self.image_list.extend(images)
    self.mask_list.extend(masks)
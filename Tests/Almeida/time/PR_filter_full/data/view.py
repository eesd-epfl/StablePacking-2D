# Stack a set of bricks of random shape
from skimage.measure import regionprops
from matplotlib.colors import ListedColormap
from matplotlib import cm
import pickle
import json
import os
import math
import cv2
from cv2 import repeat
import matplotlib.pyplot as plt
import pathlib
import math

import numpy as np
np.random.seed(2804)

# ____________________________________________________________________________parameters

# read pickle file
_root_dir = pathlib.Path(__file__).resolve().parent
_root_dir = os.path.abspath(_root_dir)
with open(_root_dir+"/base.pkl", 'rb') as f:
    final_base = pickle.load(f)


image = np.zeros(final_base.matrix.shape)
for i, stone in enumerate(final_base.placed_bricks):
    image += stone.matrix*(i+1)
#image = image*255/34
# plt.imshow(np.flip(image, axis=0), cmap=newcmp)
# plt.axis('off')
# plt.savefig(_root_dir+'/plot/rebuilt_wall.png')
cv2.imwrite(_root_dir+'/plot/rebuilt_wall.png',
            np.flip(image, axis=0).astype('uint8'))
# *******************************************
# ********Scale labels **********************
# *******************************************
back_label = 0
max_pixel = 200
min_pixel = 50
max_label = 0
min_label = np.inf
image = image.astype('uint8')
for region in regionprops(image):
    if region.label > max_label:
        max_label = region.label
    if region.label < min_label:
        min_label = region.label
# scale non-zero labels to [50,200]
image_float = image.astype(np.float32)
image_float = np.where(image_float != back_label, ((image_float-min_label) /
                                                   (max_label-min_label))*(max_pixel-min_pixel)+min_pixel, back_label)
image_scaled = image_float.astype(np.uint8)
image_scaled = np.where(image_scaled == 0, 255, image_scaled)[0:120, 0:430]
cv2.imwrite(_root_dir+'/plot/rebuilt_wall_scaled.png',
            np.flip(image_scaled, axis=0).astype('uint8'))

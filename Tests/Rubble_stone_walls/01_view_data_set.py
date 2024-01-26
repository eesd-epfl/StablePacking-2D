"""
Visualize the data set of stones.
It annotates each stone with a color id that corresponds to the clustering group that is the same as in the packing process.
Output:
    - data/pixel_id.csv: a csv file that contains the color id of each stone
    - data/original_wall_colored_by_cluster.pdf: a pdf file that contains the original wall image with color
    - data/img/cluster_False: clustering of input stones in the eccentricity-size space

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import cv2
from matplotlib import cm
from matplotlib.colors import ListedColormap
from skimage.measure import label
from skimage.measure import regionprops
from Stonepacker2D import *

tab20_cm = cm.get_cmap('tab20')
newcolors = np.concatenate([tab20_cm(np.linspace(0, 1, 20))] * 13, axis=0)
white = np.array([255/256, 255/256, 255/256, 1])
newcolors[:1, :] = white
newcmp = ListedColormap(newcolors)

#read stone wall
_root_dir = pathlib.Path(__file__).resolve().parent
_root_dir = os.path.abspath(_root_dir)
wall_width = 608
wall_height = 814
set_world_size(wall_width+2,wall_height+3)
set_dump(True, _root_dir+'/data/')
set_record_detail({'RECORD_PLACEMENT_IMG':False})
if not os.path.exists(get_dir()+f"img"):
    os.makedirs(get_dir()+f"img")
#read annotated wall image
wall_image = _root_dir+'/data/stone_wall_freepik_mask_cut.png'
wall_image = cv2.imread(wall_image, cv2.IMREAD_GRAYSCALE)
# erode
kernel = np.ones((3,3),np.uint8)
wall_image = cv2.erode(wall_image,kernel,iterations = 1)
#dilate
kernel = np.ones((3,3),np.uint8)
wall_image = cv2.dilate(wall_image,kernel,iterations = 1)

#label
wall_image = wall_image.astype(np.uint8)
wall_image = label(wall_image)
# read each region
regions = regionprops(wall_image)
# create stones
stones = []
for i, region in enumerate(regions):
    stone = Stone()
    region_label = region.label
    # skip small regions
    if region.area < 10:
        continue
    if stone.from_matrix(region.image_filled.astype(np.uint8), 1):
        stone.id = region_label
        stone = stone.rotate_min_shape_factor()['stone']
        stone.cal_shape_factor()
        stones.append(stone)
#get the color id pixel_id of each stone
pixel_id = np.zeros((len(stones),3))
for i in range(len(stones)):
    pixel_id[i,0] = stones[i].id
vendor = Vendor(stones)
nb_clusters = vendor.cluster_stones()
for i in range(len(stones)):
    pixel_id[i,1] = vendor.stones[i].cluster
    pixel_id[i,2] = vendor.stones[i].id

#write pixel_id to csv file
np.savetxt('./data/pixel_id.csv', pixel_id, delimiter=',', fmt='%d')

#plot the original wall with color id
original_wall = np.zeros((wall_image.shape))
text_kwargs = dict(ha='center', va='center',
                           fontsize=12, color='black')
fig, ax = plt.subplots()
for i in range(len(stones)):
    colored_stone = np.where(wall_image==pixel_id[i,0],int(pixel_id[i,1]+1),0)
    colored_stone = colored_stone.astype(np.uint8)
    original_wall+=colored_stone

ax.imshow(original_wall, cmap=newcmp,interpolation='none')
plt.axis('off')
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
plt.savefig("./data/original_wall_colored_by_cluster.pdf", dpi=300)
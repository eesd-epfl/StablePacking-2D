# Stack a set of bricks of random shape
import matplotlib
import os
import pathlib
import json

import cv2
from cv2 import repeat

import matplotlib.pyplot as plt

import math

import numpy as np
from Stonepacker2D import *

from skimage.measure import regionprops
def seed_everything(seed=20):
    """"
    Seed everything.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


seed_everything()

# ____________________________________________________________________________parameters
# %%
_root_dir = pathlib.Path(__file__).resolve().parent
_root_dir = os.path.abspath(_root_dir)
_rescale = 1.0
wall_file = _root_dir+'/data/Almeida_PR_wall_labels.png'
img_size = cv2.imread(wall_file).shape[0:2]
set_world_size(int(img_size[1]*_rescale)-30, int(img_size[0]*_rescale)+10)


set_dump(True, _root_dir+'/result/')

# __________________________________________________________read stones
stones = []
labels = list(range(1, 256, 1))

for label in labels:
    stone = Stone()
    if stone.from_labeled_img(label, wall_file, _rescale):
        stones.append(stone)

pixel_id = np.zeros((len(stones),3))
for i in range(len(stones)):
    pixel_id[i,0] = stones[i].id

print(f"{len(stones)} stones")
for i, stone in enumerate(stones):
    stones[i] = stone.rotate_axis_align()
    stones[i] = stones[i].rotate_min_shape_factor()

vendor = Vendor(stones)
nb_clusters = vendor.cluster_stones(plot_figure = True,plot_cluster_txt = True)


from matplotlib import cm
from matplotlib.colors import ListedColormap
tab20_cm = cm.get_cmap('tab20')
newcolors = np.concatenate([tab20_cm(np.linspace(0, 1, 20))] * 13, axis=0)
white = np.array([255/256, 255/256, 255/256, 1])
newcolors[:1, :] = white
newcmp = ListedColormap(newcolors)

for i in range(len(stones)):
    pixel_id[i,1] = vendor.stones[i].cluster
    pixel_id[i,2] = vendor.stones[i].id
wall_img = np.flip(cv2.imread(wall_file)[:, :, 0], axis=0)

# rescale image
dim_0 = int(wall_img.shape[1]*_rescale)
dim_1 = int(wall_img.shape[0]*_rescale)
dim = (dim_0, dim_1)
wall_img = cv2.resize(
    wall_img, dim, interpolation=cv2.INTER_NEAREST)
original_wall = np.zeros((wall_img.shape))
text_kwargs = dict(ha='center', va='center',
                           fontsize=12, color='black')
fig, ax = plt.subplots()
for i in range(len(stones)):
    colored_stone = np.where(wall_img==pixel_id[i,0],int(pixel_id[i,1]+1),0)
    colored_stone = colored_stone.astype(np.uint8)
    original_wall+=colored_stone
    region = regionprops(colored_stone)
    if region[0].area>100:
        ax.text(region[0].centroid[1],
                        region[0].centroid[0], f"{(pixel_id[i,2]+1):.0f}", **text_kwargs)

ax.imshow(original_wall, cmap=newcmp,interpolation='none')

plt.gca().invert_yaxis()
plt.axis('off')
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
plt.savefig(get_dir()+"img/original_wall.pdf", dpi=300)

# Stack a set of bricks of random shape
from matplotlib.colors import ListedColormap
import pickle
import json
import os
import math
import cv2
from cv2 import repeat
import matplotlib.pyplot as plt
import pathlib
import math
from matplotlib import cm
import numpy as np
from Stonepacker2D import *
np.random.seed(2804)

# ____________________________________________________________________________parameters

# read pickle file
_root_dir = pathlib.Path(__file__).resolve().parent
_root_dir = os.path.abspath(_root_dir)
with open(_root_dir+"/result/base.pkl", 'rb') as f:
    final_base = pickle.load(f)

plt_matrix = np.where((final_base.matrix != 0)&(final_base.matrix!=get_ignore_pixel_value()), (final_base.id_matrix+1), 0)
top = matplotlib.colormaps['Reds']._resample(128)
bottom = matplotlib.colormaps['Blues']._resample(128)

newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))
newcolors[0] = np.asarray([1,1,1,1])
newcmp = ListedColormap(newcolors, name='OrangeBlue')
plt.clf()
f = plt.figure(dpi=300)
f.figimage(np.flip(plt_matrix,axis = 0), cmap=newcmp,interpolation='none',resize = True,vmin = 0,vmax = 40)
plt.savefig(_root_dir+"/result/img/wall_stone_id_noscale.pdf", dpi=300)

from matplotlib import cm
from matplotlib.colors import ListedColormap
tab20_cm = cm.get_cmap('tab20')
newcolors = np.concatenate([tab20_cm(np.linspace(0, 1, 20))] * 13, axis=0)
white = np.array([255/256, 255/256, 255/256, 1])
newcolors[:1, :] = white
newcmp = ListedColormap(newcolors)

plt.close()
fig, ax = plt.subplots()
change_dict = {4:5,1:4,2:1,6:2,0:6,5:0,3:3,-1:-1}
new_cluster_matrix = np.zeros((final_base.cluster_matrix.shape))
for key, value in change_dict.items():
   new_cluster_matrix += np.where((final_base.cluster_matrix==key)&(final_base.matrix != 0)&(final_base.matrix!=get_ignore_pixel_value()),value,0)
new_cluster_matrix+=np.where((final_base.matrix==get_base_id())|(final_base.matrix==get_wedge_id()),-1,0)
plt_matrix = np.where((final_base.matrix != 0)&(final_base.matrix!=get_ignore_pixel_value()), new_cluster_matrix+1, 0)

ax.imshow(plt_matrix, cmap=newcmp,interpolation='none')
# plt_matrix = np.where((final_base.matrix != 0)&(final_base.matrix!=get_ignore_pixel_value()), final_base.cluster_matrix+1, 0)

# ax.imshow(plt_matrix, cmap=newcmp,interpolation='none')
# text_kwargs = dict(ha='center', va='center',
#                     fontsize=12, color='black')
                    

for i, stone_i in enumerate(final_base.placed_rocks):
    if stone_i.height > 10 and stone_i.width > 10:
        ax.text(final_base.rock_centers[i][0],
                final_base.rock_centers[i][1], str(stone_i.id+1), **text_kwargs)

plt.gca().invert_yaxis()
plt.axis('off')
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)

plt.savefig(_root_dir+"/result/img/wall_stone_id.pdf", dpi=300)

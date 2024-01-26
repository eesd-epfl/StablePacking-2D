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
matrix_from_img = np.flip(cv2.imread(wall_file)[:, :, 0], axis=0)
wall_width = matrix_from_img.shape[1]
wall_height = matrix_from_img.shape[0]
set_world_size(wall_width,wall_height)
set_ksteps(0)
set_dump(True, _root_dir+'/data/')
set_head_joint_state(True)
set_local_width(int(10*_rescale))
set_dilation(int(20*_rescale))
set_number_cpu(10)
set_placement_optimization("image_convolve")
set_record_time(True)
set_record_detail({'RECORD_PLACEMENT_VALUE':False,'RECORD_PLACEMENT_IMG':False,'RECORD_SELECTION_VALUE':False})
# ______________________________WEIGHTS
iw = 1
kw = 1
cw = 1
gw = 1
lw = 1
sw = {'interlocking': iw, 'kinematic': kw,
      'course_height': cw, 'global_density': gw, 'local_void': lw}
# ______________________________END WEIGHTS
set_selection_weights(sw)
matplotlib.interactive(False)
set_dump(True, _root_dir+'/result/')
if not os.path.exists(_root_dir+'/result/'):
    os.mkdir(_root_dir+'/result/')
    os.mkdir(_root_dir+'/result/img/')
    os.mkdir(_root_dir+'/result/record/')

with open(get_dir()+'img/selection.txt', 'w+') as f:
    f.write("iteration;stone id;width;x;y;score;course height;global density;interlocking;local_density;kinematic\n")


# __________________________________________________________read stones
stones = []
labels = list(range(1, 256, 1))

for label in labels:
    stone = Stone()
    if stone.from_labeled_img(label, wall_file, _rescale):
        stones.append(stone)


print(f"{len(stones)} stones")
for i, stone in enumerate(stones):
    stones[i] = stone.rotate_axis_align()
    stones[i] = stones[i].rotate_min_shape_factor()
for i, stone in enumerate(stones):
    stone.cal_shape_factor()
run_counter = 0
vendor = Vendor(stones)
nb_clusters = vendor.cluster_stones()

for key, stone in vendor.stones.items():
    cv2.imwrite(get_dir()+'img/stone_'+str(stone.id)+'_'+str(stone.cluster)+'.png', stone.matrix)
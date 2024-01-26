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
_rescale = 0.5
wall_file = _root_dir+'/data/Almeida_PR_wall_labels.png'
wall_width,wall_height = get_world_size_from_label_img(wall_file, scale=_rescale)
set_world_size(wall_width+2,wall_height+3)
set_ksteps(0)
set_dump(True, _root_dir+'/data/')
set_head_joint_state(True)
set_local_width(5)
set_dilation(20)
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
else:
    os.system('rm -rf '+_root_dir+'/result/*')
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
while(run_counter < 1):

    # # ______________________________WEIGHTS
    # iw = np.random.uniform(0, 1)
    # kw = np.random.uniform(0, 10)
    # cw = np.random.uniform(0, 1)
    # gw = np.random.uniform(0, 1)
    # sw = {'interlocking': iw, 'kinematic': kw,
    #       'course_height': cw, 'global_density': gw}
    # # ______________________________END WEIGHTS
    # set_selection_weights(sw)

    # set_dump(True, _root_dir+'/result/' +
    #          f'{iw:.2f}_{kw:.2f}_{cw:.2f}_{gw:.2f}'+'/')

    # if not os.path.exists(_root_dir+'/result/'+f'{iw:.2f}_{kw:.2f}_{cw:.2f}_{gw:.2f}'+'/'):
    #     os.mkdir(_root_dir+'/result/' +
    #              f'{iw:.2f}_{kw:.2f}_{cw:.2f}_{gw:.2f}'+'/')
    #     os.mkdir(_root_dir+'/result/' +
    #              f'{iw:.2f}_{kw:.2f}_{cw:.2f}_{gw:.2f}'+'/img/')
    #     os.mkdir(_root_dir+'/result/' +
    #              f'{iw:.2f}_{kw:.2f}_{cw:.2f}_{gw:.2f}'+'/record/')
    # else:
    #     os.system('rm -rf '+_root_dir+'/result/' +
    #               f'{iw:.2f}_{kw:.2f}_{cw:.2f}_{gw:.2f}'+'/*')
    #     os.mkdir(_root_dir+'/result/' +
    #              f'{iw:.2f}_{kw:.2f}_{cw:.2f}_{gw:.2f}'+'/img/')
    #     os.mkdir(_root_dir+'/result/' +
    #              f'{iw:.2f}_{kw:.2f}_{cw:.2f}_{gw:.2f}'+'/record/')
    # with open(get_dir()+'img/selection.txt', 'w+') as f:
    #     f.write(
    #         "iteration;stone id;width;x;y;score;course height;global density;interlocking;kinematic\n")

    base = Base()
    #ground = Brick(_wall_width, 2)
    ground = Brick(wall_width+2, 2)
    ground.id = get_base_id()
    base.add_stone(ground, [0, 0])

    base.matrix[-1, :] = get_ignore_pixel_value()
    base.matrix[:, -1] = get_ignore_pixel_value()
    base.matrix[:, 0] = get_ignore_pixel_value()
    vendor = Vendor(stones)
    nb_clusters = vendor.cluster_stones()

    final_base = build_wall_vendor(
        base, vendor, nb_clusters,vendor_type = 'full')
    # final_base = build_wall_vendor(
    #     base, vendor, nb_clusters,vendor_type = 'variant')

    plt.imshow(final_base.matrix, cmap='flag')
    # plt.axis('off')
    text_kwargs = dict(ha='center', va='center',
                       fontsize=20, color='C1')
    for i, stone_i in enumerate(final_base.placed_rocks):

        plt.text(final_base.rock_centers[i][0],
                 final_base.rock_centers[i][1], str(stone_i.id), **text_kwargs)
    plt.gca().invert_yaxis()
    result_plt_name = _root_dir+'/result/' + \
        'Base_' + \
        str(sw['interlocking'])+'_'+str(sw['kinematic'])+'_' + \
        str(sw['course_height'])+'_'+str(sw['global_density'])+'.png'
    plt.savefig(result_plt_name, dpi=300)

    evaluate_kine(final_base, save_failure=True, load='tilting_table')
    run_counter += 1

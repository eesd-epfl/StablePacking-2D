# Stack a set of bricks of random shape
import matplotlib
import os
import pathlib

import cv2
from cv2 import repeat

import matplotlib.pyplot as plt


import numpy as np
from Stonepacker2D import *
import json

def seed_everything(seed=20):
    """"
    Seed everything.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def math_operation_int(x,plus=0,times = 1):
    return int(x*times+plus)
seed_everything()

# ____________________________________________________________________________parameters
# %%
_root_dir = pathlib.Path(__file__).resolve().parent
_root_dir = os.path.abspath(_root_dir)
with open(_root_dir+'/config.json') as config_file:
    data_config = json.load(config_file)
_rescale = data_config['scale']
wall_file = _root_dir+'/data/'+data_config['input_wall_name']+'.png'
img_size = cv2.imread(wall_file).shape[0:2]
wall_width,wall_height = get_world_size_from_label_img(wall_file, scale=_rescale)
set_world_size(math_operation_int(wall_width,plus = data_config['wall_width_plus'],times = data_config['wall_width_times'])\
    ,math_operation_int(wall_height,plus = data_config['wall_height_plus'],times = data_config['wall_height_times']))
get_main_logger().critical(f"image size: {math_operation_int(wall_width,plus = data_config['wall_width_plus'],times = data_config['wall_width_times'])}, {math_operation_int(wall_height,plus = data_config['wall_height_plus'],times = data_config['wall_height_times'])}")
set_ksteps(0)
set_dump(True, _root_dir+'/data/')
set_head_joint_state(True)
set_local_width(5)
set_dilation(10)
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
        if stone.height > 3 and stone.width > 3:
            stones.append(stone)


print(f"{len(stones)} stones")
rotated_stones = []
for i, stone in enumerate(stones):
    rotate_result = stone.rotate_min_shape_factor()
    if rotate_result['success']:
        rotated_stones.append(rotate_result['stone'])
stones = rotated_stones
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
    ground = Brick(math_operation_int(wall_width,plus = data_config['wall_width_plus'],times = data_config['wall_width_times']),2)
    ground.id = get_base_id()
    base.add_stone(ground, [0, 0])


    base.matrix[-1, :] = get_ignore_pixel_value()
    base.matrix[:, -1] = get_ignore_pixel_value()
    base.matrix[:, 0] = get_ignore_pixel_value()
    vendor = Vendor(stones)
    nb_clusters = vendor.cluster_stones()

    # final_base = build_wall_vendor(
    #     base, vendor, int(1.3*nb_clusters))
    final_base = build_wall_vendor(
        base, vendor, nb_clusters,vendor_type = data_config['vendor_type'],construction_order=data_config['construction_order'],variant_sample =data_config['variant_sample'],allow_cutting = data_config['allow_cutting'])

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

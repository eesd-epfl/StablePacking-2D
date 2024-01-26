"""
Main script for packing rubbles to build a wall
"""
import matplotlib
import os
import pathlib
import datetime
import cv2
from cv2 import repeat

import matplotlib.pyplot as plt

import math

import numpy as np
from Stonepacker2D import *
from skimage.measure import label
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
_rescale = 0.5
wall_width = 608
wall_height = 608
set_world_size(int(wall_width*_rescale)+2,int(wall_height*_rescale)+3)
set_ksteps(0)
set_stabalize_method('rotation')
set_head_joint_state(True)
set_local_width(10)
set_dilation(20)
set_number_cpu(10)
set_placement_optimization("image_convolve")
set_record_detail({'RECORD_PLACEMENT_IMG':False,'RECORD_PLACEMENT_VALUE':False})

matplotlib.interactive(False)
time_stamp= datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if not os.path.exists(_root_dir+'/result_'+time_stamp+'/'):
    os.mkdir(_root_dir+'/result_'+time_stamp+'/')

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
initial_rotation_angles = dict()  # positive clockwise
for i, region in enumerate(regions):
    stone = Stone()
    region_label = region.label
    # skip small regions
    if region.area < 10:
        continue
    if stone.from_matrix(region.image_filled.astype(np.uint8), _rescale):
        # plt.imshow(region.image)
        # plt.title(str(region_label))
        # plt.show()
        stone.id = region_label
        rotation_result = stone.rotate_min_shape_factor()
        stone = rotation_result['stone']
        if stone is None:
            plt.imshow(region.image)
            plt.title(str(region_label))
            plt.show()
        initial_rotation_angles[stone.id] = rotation_result['angle']
        stone.cal_shape_factor()
        stones.append(stone)
        # plt.imshow(regionprops(stone.matrix.astype(np.uint8))[0].image)
        # plt.title(str(region_label))
        # plt.show()



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
max_nb_runs = 20
for wall_i in range(max_nb_runs):
    set_dump(True, _root_dir+'/result_'+time_stamp+f'_wall{wall_i}'+'/')
    if not os.path.exists(_root_dir+'/result_'+time_stamp+f'_wall{wall_i}'+'/'):
        os.mkdir(_root_dir+'/result_'+time_stamp+f'_wall{wall_i}'+'/')
        os.mkdir(_root_dir+'/result_'+time_stamp+f'_wall{wall_i}'+'/img/')
        os.mkdir(_root_dir+'/result_'+time_stamp+f'_wall{wall_i}'+'/record/')
    else:
        os.system('rm -rf '+_root_dir+'/result_'+time_stamp+f'_wall{wall_i}'+'/*')
        os.mkdir(_root_dir+'/result_'+time_stamp+f'_wall{wall_i}'+'/img/')
        os.mkdir(_root_dir+'/result_'+time_stamp+f'_wall{wall_i}'+'/record/')
    with open(get_dir()+'img/selection.txt', 'w+') as f:
        f.write("iteration;stone id;width;x;y;score;course height;global density;interlocking;local_density;kinematic\n")
    save_transformation_dir = get_dir()
    with open(save_transformation_dir+'transformation.txt', 'w+') as f:
        f.write("id;d_x;d_y;angle\n")


    base = Base()
    ground = Brick(wall_width+2, 2)
    ground.id = get_base_id()
    base.add_stone(ground, [0, 0])
    #base.matrix[:, -15:-1] = 1
    base.matrix[:, -1] = get_ignore_pixel_value()
    base.matrix[:, 0] = get_ignore_pixel_value()
    base.matrix[-1, :] = get_ignore_pixel_value()
    vendor = Vendor(stones,False)
    nb_clusters = vendor.cluster_stones()
    final_base = build_wall_vendor(
        base, vendor, nb_clusters,vendor_type = 'variant')

    # Output transformation
    for placed_stone in final_base.placed_rocks:
        current_stone_matrix = np.where(final_base.id_matrix==placed_stone.id,1,0)
        stone_prop = regionprops(current_stone_matrix.astype(np.uint8))
        stone_center = stone_prop[0].centroid
        d_x = (stone_center[1])/_rescale
        d_y = (stone_center[0])/_rescale
        angle_d = placed_stone.rot_angle
        angle_i = placed_stone.rotate_from_ori
        stone_id = placed_stone.id
        print(
            f"stone {stone_id} is placed at ({d_x},{d_y}) with stable angle {angle_d} degree")
        with open(save_transformation_dir+'transformation.txt', 'a+') as f:
            f.write("{};{};{};{}\n".format(placed_stone.id, d_x, d_y,
                    initial_rotation_angles[placed_stone.id]+angle_i+angle_d))

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import matplotlib.cm as cm
    tab20_cm = cm.get_cmap('tab20')
    newcolors = np.concatenate([tab20_cm(np.linspace(0, 1, 20))] * 13, axis=0)
    white = np.array([255/256, 255/256, 255/256, 1])
    newcolors[:1, :] = white
    newcmp = ListedColormap(newcolors)

    plt.imshow(final_base.matrix, cmap=newcmp,interpolation='none')
    # plt.axis('off')
    text_kwargs = dict(ha='center', va='center',
                        fontsize=10, color='black')
    for i, stone_i in enumerate(final_base.placed_rocks):
        plt.text(final_base.rock_centers[i][0],
                    final_base.rock_centers[i][1], str(i+1), **text_kwargs)
    plt.gca().invert_yaxis()
    result_plt_name = _root_dir+'/result_'+time_stamp+f'_wall{wall_i}'+'/' + \
        'Base_' + \
        str(sw['interlocking'])+'_'+str(sw['kinematic'])+'_' + \
        str(sw['course_height'])+'_'+str(sw['global_density'])+'.png'
    plt.savefig(result_plt_name, dpi=300)

    kinematics_ = evaluate_kine(final_base, save_failure=True, load='tilting_table')




"""
Apply transformation result to the original stone images to obtain high resolution wall images.

Input:
    - stone_cluster_id: the pixel_id.csv file generated from 00_view_data_set.py
Output:
    - High resolution wall images with stones, in each result folder, starting with "base_rebuilt***"
"""

import os
import cv2
import numpy as np
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from skimage.measure import label
from Stonepacker2D import *
from matplotlib import cm
from matplotlib.colors import ListedColormap
import pathlib

tab20_cm = cm.get_cmap('tab20')
newcolors = np.concatenate([tab20_cm(np.linspace(0, 1, 20))] * 13, axis=0)
white = np.array([255/256, 255/256, 255/256, 1])
newcolors[:1, :] = white
newcmp = ListedColormap(newcolors)
data_dir_prefix = "./result_20240131-113837_wall"
_rescale = 0.5
wall_width = 608
wall_height = 608
wall_size = (wall_width, wall_height)
set_world_size(int(wall_width*_rescale)+2,int(wall_height*_rescale)+3)



# read cluster stored in csv to numpy array
# region_label, cluster, id
stone_cluster_id = np.genfromtxt('./data/pixel_id.csv', delimiter=',')
stone_cluster_id_dict = {}
for i in range(stone_cluster_id.shape[0]):
    stone_cluster_id_dict[int(stone_cluster_id[i, 0])] = int(
        stone_cluster_id[i, 1])

#read stones
_root_dir = pathlib.Path(__file__).resolve().parent
_root_dir = os.path.abspath(_root_dir)
set_dump(True, _root_dir+'/data/')
set_record_detail({'RECORD_PLACEMENT_IMG':False})
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
stone_images_original = dict()
stones = []
initial_rotation_angles = dict()  # positive clockwise
for i, region in enumerate(regions):
    stone = Stone()
    region_label = region.label
    # skip small regions
    if region.area < 10:
        continue
    if stone.from_matrix(region.image_filled.astype(np.uint8), _rescale):
        stone.id = region_label
        stone_matrix = np.zeros_like(wall_image)#region.image_filled.astype(np.uint8)
        stone_matrix[region.bbox[0]:region.bbox[2],region.bbox[1]:region.bbox[3]] = region.image_filled.astype(np.uint8)
        stone_images_original[region_label] = stone_matrix

#move stones to center of image
for key in stone_images_original.keys():
    region = regionprops(stone_images_original[key].astype(np.uint8))
    center = region[0].centroid#row, column
    image_center = (stone_images_original[key].shape[0]//2, stone_images_original[key].shape[1]//2)
    M = np.float32([[1, 0, image_center[1]-center[1]], [0, 1, image_center[0]-center[0]]])
    stone_images_original[key] = cv2.warpAffine(stone_images_original[key], M, (
        stone_images_original[key].shape[1], stone_images_original[key].shape[0]), flags=cv2.WARP_FILL_OUTLIERS)

for i in range(12):
    data_dir = data_dir_prefix+f'{i}/'
    stone_images = stone_images_original.copy()
    # read transformation.txt
    # stone_id, d_x, d_y, angle_d
    stone_transformation = {}
    placement_sequence = {}
    with open(data_dir+'transformation.txt', 'r') as f:
        lines = f.readlines()
        for line_i,line in enumerate(lines[1:]):
            line = line.split(';')
            stone_transformation[int(line[0])] = [float(
                line[1]), float(line[2]), float(line[3])]
            placement_sequence[int(line[0])] = int(line_i)

    # move stone images to base
    plt.close()
    base_image = np.zeros((wall_size[1],wall_size[0]), dtype=np.uint8)
    base_image_colored = np.zeros_like(base_image)
    base_image_clustered_stones = np.zeros_like(base_image)
    #couter for overlapp
    overlap_counter = 0
    for seq_i, stone_id in enumerate(stone_transformation.keys()):
        # rotate
        angle = stone_transformation[stone_id][2]
        center = (stone_images[stone_id].shape[1]//2,
                stone_images[stone_id].shape[0]//2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # print("Rotation angle: ", angle)
        stone_images[stone_id] = cv2.warpAffine(stone_images[stone_id], M, (
            stone_images[stone_id].shape[1], stone_images[stone_id].shape[0]), flags=cv2.WARP_FILL_OUTLIERS)

        # move to (p_x, p_y)
        p_x = stone_transformation[stone_id][0]
        p_y = stone_transformation[stone_id][1]
        #plt.scatter(p_x, p_y, c='g', s=10)
        # find center of stone using regionprops
        stone_prop = regionprops(stone_images[stone_id].astype(np.uint8))
        stone_r_c = stone_prop[0].centroid
        stone_center = (stone_r_c[1], stone_r_c[0])
        # move stone by (p_x, p_y) - stone_center
        M = np.float32([[1, 0, p_x-stone_center[0]], [0, 1, p_y-stone_center[1]]])
        stone_images[stone_id] = cv2.warpAffine(stone_images[stone_id], M, (
            stone_images[stone_id].shape[1], stone_images[stone_id].shape[0]), flags=cv2.WARP_FILL_OUTLIERS)
        # add stone to base
        #crop stone image to the same size of base image
        stone_images[stone_id] = stone_images[stone_id][0:base_image.shape[0],0:base_image.shape[1]]
        #add zeros to the stone image to match the size of base image
        stone_images[stone_id] = np.pad(stone_images[stone_id], ((0,base_image.shape[0]-stone_images[stone_id].shape[0]),(0,base_image.shape[1]-stone_images[stone_id].shape[1])), 'constant', constant_values=(0,0))
        # check overlapping
        if len(np.argwhere((stone_images[stone_id] != 0)&(base_image!=0)))>=200:
            print("overlapping")
            overlap_counter += 1
            continue


        # save the transformed stones image
        if os.path.exists(data_dir+'transformed_stones/')==False:
            os.mkdir(data_dir+'transformed_stones/')
        cv2.imwrite(data_dir+'transformed_stones/iteration_'+str(seq_i)+'_stone_'+str(stone_id)+'.png',stone_images[stone_id])
        
        base_image = np.where(
            stone_images[stone_id] != 0, stone_images[stone_id], base_image)
        base_image_colored = np.where(
            stone_images[stone_id] != 0, seq_i+1, base_image_colored)
        base_image_clustered_stones = np.where(
            (stone_images[stone_id] != 0) & (base_image_clustered_stones==0), stone_cluster_id_dict[stone_id]+1, base_image_clustered_stones)
        
        stone_prop = regionprops(stone_images[stone_id].astype(np.uint8))
        stone_r_c = stone_prop[0].centroid
        stone_center = (stone_r_c[1], stone_r_c[0])
        #plt.scatter(stone_center[0], stone_center[1], c='r', s=10)

        text_kwargs = dict(ha='center', va='center',
                        fontsize=10, color='black')
        if stone_prop[0].area>500:
            plt.text(p_x, stone_transformation[stone_id]
                    [1], str(placement_sequence[stone_id]+1-overlap_counter), **text_kwargs)
            #print(placement_sequence[stone_id]+1)


    plt.imshow(base_image_clustered_stones, cmap=newcmp,interpolation='none')
    plt.gca().invert_yaxis()
    #remove axis
    plt.axis('off')
    result_plt_name = data_dir+'base_rebuilt_with_clustered_stones.pdf'
    plt.savefig(result_plt_name, dpi=300,transparent=True)

    plt.close()
    plt.imshow(base_image, cmap=newcmp, alpha=1,interpolation='none')
    plt.gca().invert_yaxis()
    #remove axis
    plt.axis('off')
    result_plt_name = data_dir+'base_rebuilt_with_annotate.png'
    plt.savefig(result_plt_name, dpi=300,transparent=True)
    cv2.imwrite(data_dir+'stacked_wall_label.png',base_image_colored)
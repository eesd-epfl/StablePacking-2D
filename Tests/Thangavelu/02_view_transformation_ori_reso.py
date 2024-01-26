
"""
Apply transformation result to the original stone images to obtain high resolution wall images.

Input (switching between datasets by commenting/uncommenting the corresponding lines):
    - data_dir_prefix: the prefix of the directory that contains the transformation.txt file
    - noise_level: the noise level of the stone set
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
def get_wall_size_from_image(wall_image,rescale):
    #label
    wall_image = wall_image.astype(np.uint8)
    wall_image = label(wall_image)
    # read each region
    regions = regionprops(wall_image)
    # get average bounding box width and height of regions
    average_region_width = 0
    average_region_height = 0
    nb_valid_regions = 0
    for region in regions:
        if region.area < 50:
            continue
        max_width = region.bbox[3]-region.bbox[1]
        max_height = region.bbox[2]-region.bbox[0]
        average_region_width += max_width
        average_region_height += max_height
        nb_valid_regions += 1
    average_region_width /= nb_valid_regions
    average_region_height /= nb_valid_regions
    wall_width = average_region_width*5
    wall_height = average_region_height*5
    return (int(wall_width+2/rescale), int(wall_height+3/rescale))
tab20_cm = cm.get_cmap('tab20')
newcolors = np.concatenate([tab20_cm(np.linspace(0, 1, 20))] * 13, axis=0)
white = np.array([255/256, 255/256, 255/256, 1])
newcolors[:1, :] = white
newcmp = ListedColormap(newcolors)
"""
Configurations
"""
#----------------------------------------------------noise 20
data_dir_prefix = "./result_20231114-155912_wall"
noise_level = 20
# #----------------------------------------------------noise 10
# data_dir_prefix = "./result_20231114-172513_wall"
# noise_level = 10
# #----------------------------------------------------noise 30
# data_dir_prefix = "./result_20231114-194209_wall"
# noise_level = 30
# #----------------------------------------------------noise 40
# data_dir_prefix = "./result_20231115-095452_wall"
# noise_level = 40
# #----------------------------------------------------noise 50
# data_dir_prefix = "./result_20231115-130605_wall"
# noise_level = 50
# #----------------------------------------------------noise 60
# data_dir_prefix = "./result_20231115-151823_wall"
# noise_level = 60
# #----------------------------------------------------noise 70
# data_dir_prefix = "./result_20231115-174925_wall"
# noise_level = 70
# #----------------------------------------------------noise 80
# data_dir_prefix = "./result_20231115-231235_wall"
# noise_level = 80

# # ----------------------------------------------------noise 80 with interval 10
# data_dir_prefix = "./result_20231120-223433_wall"
# noise_level = 80
#----------------------------------------------------noise 70 with interval 10
data_dir_prefix = "./result_20231127-171331_wall"
noise_level = 70
#----------------------------------------------------noise 60 with interval 10
data_dir_prefix = "./result_20231128-101035_wall"
noise_level = 60
#----------------------------------------------------noise 50 with interval 10
data_dir_prefix = "./result_20231128-172054_wall"
noise_level = 50
#----------------------------------------------------noise 40 with interval 10
data_dir_prefix = "./result_20231129-082758_wall"
noise_level = 40
#----------------------------------------------------noise 30 with interval 10
data_dir_prefix = "./result_20231129-110402_wall"
noise_level = 30
#----------------------------------------------------noise 20 with interval 10
data_dir_prefix = "./result_20231129-131421_wall"
noise_level = 20
#----------------------------------------------------noise 10 with interval 10
data_dir_prefix = "./result_20231129-150138_wall"
noise_level = 10
wall_image = f'./data/stones_noise_{noise_level}_labels.png'




_rescale = 0.5
wall_image = cv2.imread(wall_image, cv2.IMREAD_GRAYSCALE)
wall_size = get_wall_size_from_image(wall_image,_rescale)
wall_height = int(wall_size[1]*_rescale)
wall_width = int(wall_size[0]*_rescale)
for i in range(5):
    data_dir = data_dir_prefix+str(i)+'/'
    # read base image
    base_image = np.zeros((wall_size[1],wall_size[0]), dtype=np.uint8)
    # read transformation.txt
    # stone_id, d_x, d_y, angle_d
    stone_transformation = {}
    with open(data_dir+'transformation.txt', 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.split(';')
            stone_transformation[int(line[0])] = [float(
                line[1]), float(line[2]), float(line[3])]
    stone_images = dict()
    #read annotated wall image
    set_world_size(wall_width+2,wall_height+3)
    _root_dir = pathlib.Path(__file__).resolve().parent
    _root_dir = os.path.abspath(_root_dir)
    set_dump(True, _root_dir+'/data/')
    set_record_detail({'RECORD_PLACEMENT_IMG':False})

    wall_image = f'./data/stones_noise_{noise_level}_labels.png'
    wall_image = cv2.imread(wall_image, cv2.IMREAD_GRAYSCALE)

    #label
    wall_image = wall_image.astype(np.uint8)
    wall_image = label(wall_image)

    # read each region
    regions = regionprops(wall_image)
    for i, region in enumerate(regions):
        stone = Stone()
        region_label = region.label
        # skip small regions
        if region.area < 50:
            continue
        stone_matrix = np.zeros_like(wall_image)#region.image_filled.astype(np.uint8)
        stone_matrix[region.bbox[0]:region.bbox[2],region.bbox[1]:region.bbox[3]] = region.image_filled.astype(np.uint8)
        stone_images[region_label] = stone_matrix
    
    for key in stone_images.keys():
        region = regionprops(stone_images[key].astype(np.uint8))
        center = region[0].centroid#row, column
        image_center = (stone_images[key].shape[0]//2, stone_images[key].shape[1]//2)
        M = np.float32([[1, 0, image_center[1]-center[1]], [0, 1, image_center[0]-center[0]]])
        stone_images[key] = cv2.warpAffine(stone_images[key], M, (
            stone_images[key].shape[1], stone_images[key].shape[0]), flags=cv2.WARP_FILL_OUTLIERS)

    # move stone images to base
    plt.close()
    base_image_colored = np.zeros_like(base_image)
    for seq_i, stone_id in enumerate(stone_transformation.keys()):
        stone_id_in_image = stone_id%1000
        # rotate
        angle = stone_transformation[stone_id][2]
        center = (stone_images[stone_id_in_image].shape[1]//2,
                stone_images[stone_id_in_image].shape[0]//2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # print("Rotation angle: ", angle)
        stone_images[stone_id_in_image] = cv2.warpAffine(stone_images[stone_id_in_image], M, (
            stone_images[stone_id_in_image].shape[1], stone_images[stone_id_in_image].shape[0]), flags=cv2.WARP_FILL_OUTLIERS)

        # move to (p_x, p_y)
        p_x = stone_transformation[stone_id][0]
        p_y = stone_transformation[stone_id][1]
        #plt.scatter(p_x, p_y, c='g', s=10)
        # find center of stone using regionprops
        stone_prop = regionprops(stone_images[stone_id_in_image].astype(np.uint8))
        stone_r_c = stone_prop[0].centroid
        stone_center = (stone_r_c[1], stone_r_c[0])
        # move stone by (p_x, p_y) - stone_center
        M = np.float32([[1, 0, p_x-stone_center[0]], [0, 1, p_y-stone_center[1]]])
        stone_images[stone_id_in_image] = cv2.warpAffine(stone_images[stone_id_in_image], M, (
            stone_images[stone_id_in_image].shape[1], stone_images[stone_id_in_image].shape[0]), flags=cv2.WARP_FILL_OUTLIERS)
        # add stone to base
        #crop stone image to the same size of base image
        stone_images[stone_id_in_image] = stone_images[stone_id_in_image][0:base_image.shape[0],0:base_image.shape[1]]
        #add zeros to the stone image to match the size of base image
        stone_images[stone_id_in_image] = np.pad(stone_images[stone_id_in_image], ((0,base_image.shape[0]-stone_images[stone_id_in_image].shape[0]),(0,base_image.shape[1]-stone_images[stone_id_in_image].shape[1])), 'constant', constant_values=(0,0))
        # check overlapping
        if len(np.argwhere((stone_images[stone_id_in_image] != 0)&(base_image!=0)))>=200:
            print("overlapping")
            continue
        
        base_image = np.where(
            stone_images[stone_id_in_image] != 0, stone_images[stone_id_in_image], base_image)
        base_image_colored = np.where(
            stone_images[stone_id_in_image] != 0, seq_i+1, base_image_colored)
        stone_prop = regionprops(stone_images[stone_id_in_image].astype(np.uint8))
        stone_r_c = stone_prop[0].centroid
        stone_center = (stone_r_c[1], stone_r_c[0])
        #plt.scatter(stone_center[0], stone_center[1], c='r', s=10)

        text_kwargs = dict(ha='center', va='center',
                        fontsize=14, color='black')
        plt.text(p_x, stone_transformation[stone_id]
                [1], str(seq_i+1), **text_kwargs)
    # #remove borders of base image where elements are zero
    # base_image = base_image[~np.all(base_image == 0, axis=1)]
    # base_image = base_image[:, ~np.all(base_image == 0, axis=0)]
    # base_image_colored = base_image_colored[~np.all(base_image_colored == 0, axis=1)]
    # base_image_colored = base_image_colored[:, ~np.all(base_image_colored == 0, axis=0)]
    none_image = np.zeros_like(base_image)
    plt.imshow(none_image,alpha = 0)
    plt.axis('off')
    plt.gca().invert_yaxis()
    plt.savefig(data_dir+'construction_sequence', dpi=300,transparent=True)
    plt.close()
    plt.imshow(base_image, cmap=newcmp, alpha=1,interpolation='none')
    plt.gca().invert_yaxis()
    #remove axis
    plt.axis('off')
    result_plt_name = data_dir+'base_rebuilt_with_annotate.png'
    plt.savefig(result_plt_name, dpi=300,transparent=True)
    cv2.imwrite(data_dir+'stacked_wall_label.png',base_image_colored)


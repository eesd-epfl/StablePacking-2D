import math
import glob
from PIL import Image
from skimage.measure import regionprops
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
# filepaths
fp_in = "./result_20240126-170905_wall0/transformed_stones/*.png"
fp_out = "./result_20240126-170905_wall0/transformed_stones/build_process.gif"

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif


def get_iteration_number(file):
    return int(file.split("_")[-3])

# read all stone images
imgs = [Image.open(f)
        for f in sorted(glob.glob(fp_in), key=get_iteration_number)]
nb_stones = len(imgs)
print("There are {} stones in the dataset".format(nb_stones))

# create gif image
nb_rows = math.ceil(np.sqrt(nb_stones))
nb_rows = 4
nb_cols = math.ceil(nb_stones/nb_rows)
col_interval = 150
row_interval = 150
shape_stoneset = (row_interval*nb_rows, col_interval*nb_cols)
stone_dataset = np.zeros(shape_stoneset)


stone_dataset_timeline = []
random_order = np.arange(nb_stones)
np.random.shuffle(random_order)
for i, image in enumerate(imgs):
    # *******************************************
    # ********Scale labels **********************
    # *******************************************
    back_label = 0
    max_pixel = int((i/len(imgs))*(255-100)+100)
    min_pixel = int((i/len(imgs))*(255-100)+100)-1
    max_label = 1
    min_label = 0
    image = np.asarray(image.convert('L'))

    image_float = image.astype(np.float32)
    image_float = np.where(image_float != back_label, ((image_float-min_label) /
                                                       (max_label-min_label))*(max_pixel-min_pixel)+min_pixel, back_label)
    stone = np.flip(image_float.astype(np.uint8), axis=0)
    # move to center, rotate randomly, move to origin, move to random place

    random_angle = np.random.uniform()
    stone_prop = regionprops(stone)[0]
    stone_center = (stone_prop.centroid[1],
                    stone_prop.centroid[0])
    T = np.float32([[1, 0, shape_stoneset[1]/2-stone_center[0]],
                    [0, 1, shape_stoneset[0]/2-stone_center[1]]])
    stone = cv2.warpAffine(
        stone, T, stone_dataset.shape[1::-1], flags=cv2.WARP_FILL_OUTLIERS)

    rot_mat = cv2.getRotationMatrix2D(
        (shape_stoneset[1]/2, shape_stoneset[0]/2), random_angle*40-20, 1.0)
    rotated_stone_pixels = cv2.warpAffine(
        stone, rot_mat, stone_dataset.shape[1::-1], flags=cv2.WARP_FILL_OUTLIERS)
    # plt.imshow(stone)
    # plt.show()
    # plt.imshow(rotated_stone_pixels)
    # plt.show()
    stone_prop = regionprops(rotated_stone_pixels)[0]
    origin = (stone_prop.bbox[0], stone_prop.bbox[1])
    trans_col = col_interval*((random_order[i]-1) % nb_cols)-origin[1]
    trans_row = row_interval * \
        (math.floor((random_order[i]-1) / nb_cols))-origin[0]
    T = np.float32([[1, 0, trans_col],
                    [0, 1, trans_row]])
    translated_stone_pixels = cv2.warpAffine(
        rotated_stone_pixels, T, stone_dataset.shape[1::-1], flags=cv2.WARP_FILL_OUTLIERS)
    # plt.imshow(translated_stone_pixels)
    # plt.show()
    while np.multiply(translated_stone_pixels, stone_dataset).sum() > 0:
        T = np.float32([[1, 0, trans_col],
                        [0, 1, -1]])
        translated_stone_pixels = cv2.warpAffine(
            translated_stone_pixels, T, stone_dataset.shape[1::-1], flags=cv2.WARP_FILL_OUTLIERS)
    stone_dataset += translated_stone_pixels
    stone_dataset_timeline.append(Image.fromarray(stone_dataset))
stone_dataset_timeline.reverse()
Image.fromarray(stone_dataset).save(fp=fp_out, format='GIF', append_images=stone_dataset_timeline,
                                    save_all=True, duration=600, loop=0)

# # combine wall and data set row by row
# shape_all = (np.asarray(imgs[0].convert('L')).shape[0]+shape_stoneset[0]+20,
#              max(np.asarray(imgs[0].convert('L')).shape[1], shape_stoneset[1]))

# combine wall and data set column by column
_left_margin = 100
between_wall_dataset = 100
shape_all = (max(np.asarray(imgs[0].convert('L')).shape[0],shape_stoneset[0]),
             _left_margin+between_wall_dataset+np.asarray(imgs[0].convert('L')).shape[1]+shape_stoneset[1])

shape_wall = np.asarray(imgs[0].convert('L')).shape
built_wall = np.zeros(shape_wall)
dataset_subtract_all_stone = stone_dataset
construction_with_dataset = []
for i, image in enumerate(imgs):
    built_wall_withdata_set = np.zeros(shape_all)
    # *******************************************
    # ********Scale labels **********************
    # *******************************************
    back_label = 0
    max_pixel = int((i/len(imgs))*(255-100)+100)
    min_pixel = int((i/len(imgs))*(255-100)+100)-1
    max_label = 1
    min_label = 0
    image = np.asarray(image.convert('L'))

    image_float = image.astype(np.float32)
    image_float = np.where(image_float != back_label, ((image_float-min_label) /
                                                       (max_label-min_label))*(max_pixel-min_pixel)+min_pixel, back_label)
    built_wall += np.flip(image_float.astype(np.uint8), axis=0)
    
    if built_wall.shape[0]<built_wall_withdata_set.shape[0]:
        start_row_built_wall = np.floor((built_wall_withdata_set.shape[0]-built_wall.shape[0])/2).astype(np.int)
    else:
        start_row_built_wall = 0
    built_wall_withdata_set[start_row_built_wall:start_row_built_wall+shape_wall[0],
                            _left_margin:_left_margin+shape_wall[1]] = built_wall
    
    

    selected_color = 199
    unselected_color = 45
    dataset_subtract_all_stone = np.where(dataset_subtract_all_stone == regionprops(image_float.astype(np.uint8))[
        0].label, selected_color, dataset_subtract_all_stone)
    dataset_subtract_all_stone_color = np.where((dataset_subtract_all_stone == 0) | (
        dataset_subtract_all_stone == selected_color), dataset_subtract_all_stone, unselected_color)

    # # combine wall and data set row by row
    # built_wall_withdata_set[shape_wall[0]+20:,
    #                         :] = dataset_subtract_all_stone_color
    
    #combine wall and data set column by column
    if dataset_subtract_all_stone_color.shape[0]<built_wall_withdata_set.shape[0]:
        start_row_dataset = np.floor((built_wall_withdata_set.shape[0]-dataset_subtract_all_stone_color.shape[0])/2).astype(np.int)
    else:
        start_row_dataset = 0
    built_wall_withdata_set[start_row_dataset:start_row_dataset+dataset_subtract_all_stone_color.shape[0],shape_wall[1]+between_wall_dataset+_left_margin:] = dataset_subtract_all_stone_color

    # set background to white
    built_wall_withdata_set = np.where(
        built_wall_withdata_set == 0, 255, built_wall_withdata_set)
    
    #add left and right black bound
    bound_thickness = 5
    built_wall_withdata_set[start_row_built_wall+bound_thickness:start_row_built_wall+shape_wall[0],_left_margin-bound_thickness:_left_margin] = 0
    built_wall_withdata_set[start_row_built_wall+bound_thickness:start_row_built_wall+shape_wall[0],_left_margin+shape_wall[1]:_left_margin+shape_wall[1]+bound_thickness] = 0
    built_wall_withdata_set[start_row_built_wall+shape_wall[0]-bound_thickness:start_row_built_wall+shape_wall[0],_left_margin:_left_margin+shape_wall[1]] = 0

    construction_with_dataset.append(Image.fromarray(built_wall_withdata_set))
construction_with_dataset[0].save(fp=fp_out, format='GIF', append_images=construction_with_dataset,
                                  save_all=True, duration=600, loop=0)

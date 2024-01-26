

import numpy as np
from ..utils.constant import get_record_detail,get_placement_optimization,get_local_width, get_ignore_pixel_value, get_stabalize_method, get_dir
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from skimage.measure import regionprops
INTERLOCKING_PIXEL = 3
NUMBER_OF_PIXELS_AS_EPSILON = 3

def get_void_to_left(base,stone):
    m_shape = base.matrix.shape
    #distance to the left
    column_index = np.tile(np.arange(m_shape[1])+1,(m_shape[0],1))
    base_matrix_effective = np.where(base.matrix != get_ignore_pixel_value(), base.matrix, 0)
    column_index_base  = np.where(base_matrix_effective!=0,column_index,0)
    column_index_stone = np.where(stone.matrix!=0,column_index,np.inf)
    mask_l = np.zeros(m_shape)
    mask_l[:,0:int(stone.center[0])] = 1
    column_index_base_masked = np.multiply(column_index_base,mask_l)
    base_rb = np.max(column_index_base_masked,axis = 1)
    stone_lb = np.min(column_index_stone,axis = 1)
    distance_l = stone_lb-base_rb-1
    # print("base_rb",base_rb)
    # print("stone lb",stone_lb)
    # print("distance",distance_l)
    distance_positive_sum_l = np.sum(distance_l[(distance_l>0)&(distance_l<np.inf)])

    #distance to the floor
    row_index = np.tile(np.arange(m_shape[0])+1,(m_shape[1],1)).T
    row_index_base  = np.where(base_matrix_effective!=0,row_index,0)
    row_index_stone = np.where(stone.matrix!=0,row_index,np.inf)
    mask_b = np.zeros(m_shape)
    mask_b[0:int(stone.center[1]),:] = 1
    row_index_base_masked = np.multiply(row_index_base,mask_b)
    base_tb = np.max(row_index_base_masked,axis = 0)
    stone_bb = np.min(row_index_stone,axis = 0)
    distance_b = stone_bb-base_tb-1
    distance_positive_sum_b = np.sum(distance_b[(distance_b>0)&(distance_b<np.inf)])

    #void_ratio = 0.5*distance_positive_sum_l/(m_shape[1]*stone.height)+0.5*distance_positive_sum_b/(m_shape[0]*stone.width)
    void_ratio = (distance_positive_sum_l+distance_positive_sum_b)/(m_shape[1]*stone.height+m_shape[0]*stone.width)
    return void_ratio
def evaluate(base, stone):
    """Evaluate a position of a stone

    :param base: Current base with placed stones
    :type base: Stonepacker2D.base.Base
    :param stone: Stone placed on the position
    :type stone: Stonepacker2D.stone.Stone
    :return: Distance to be minimized
    :rtype: float
    """
    write_txt = get_record_detail()['RECORD_PLACEMENT_VALUE']
    # check penetration
    product = np.multiply(base.matrix, stone.matrix)
    # plt.clf()
    # plt.imshow(stone.matrix)
    # plt.show()
    # plt.clf()
    # plt.imshow(base.matrix)
    # plt.show()
    # plt.clf()

    if np.any(product):
        # write to txt
        if write_txt == True:
            with open(get_dir()+f"record/place_ite{len(base.placed_bricks)}_brick{stone.id}_{stone.width}_{stone.rotate_from_ori}.txt", "a+") as f:
                f.write("{};{};{};{};{};{}\n".format(stone.center[0], stone.center[1],
                        100, 0, 0,0))
        return float("+inf"),-1,-1
    if len(np.nonzero(stone.matrix)) == 0 or len(np.nonzero(stone.matrix)[0]) == 0:
        if write_txt == True:
            with open(get_dir()+f"record/place_ite{len(base.placed_bricks)}_brick{stone.id}_{stone.width}_{stone.rotate_from_ori}.txt", "a+") as f:
                f.write("{};{};{};{};{};{}\n".format(stone.center[0], stone.center[1],
                        100, 0, 0,0))
        return float("+inf"),-1,-1
    distance = 0
    # _____________________________________________________________use local void
    _base_after = base.matrix+stone.matrix
    local_width = get_local_width()
    if local_width == 0:
        local_width_w = stone.height
        local_width_h = stone.width
    else:
        local_width_w = local_width
        local_width_h = local_width
    if get_placement_optimization() != "image_convolve":        
        local = _base_after[max(0, int(stone.center[1]-stone.height/2-local_width_h)):min(stone.matrix.shape[0], int(stone.center[1] + stone.height/2)),
                            max(0, int(stone.center[0]-stone.width/2-local_width_w)):min(stone.matrix.shape[1], int(stone.center[0]+stone.width/2))]

        nb_void = len(np.argwhere(local == 0))
        void_ratio = nb_void/((stone.height+local_width_h)*(stone.width+local_width_w)-stone.height*stone.width)
    else:
        m_shape = base.matrix.shape
        #distance to the left
        column_index = np.tile(np.arange(m_shape[1])+1,(m_shape[0],1))
        base_matrix_effective = np.where(base.matrix != get_ignore_pixel_value(), base.matrix, 0)
        column_index_base  = np.where(base_matrix_effective!=0,column_index,0)
        column_index_stone = np.where(stone.matrix!=0,column_index,np.inf)
        mask_l = np.zeros(m_shape)
        mask_l[:,0:int(stone.center[0])] = 1
        column_index_base_masked = np.multiply(column_index_base,mask_l)
        base_rb = np.max(column_index_base_masked,axis = 1)
        stone_lb = np.min(column_index_stone,axis = 1)
        distance_l = stone_lb-base_rb-1
        # print("base_rb",base_rb)
        # print("stone lb",stone_lb)
        # print("distance",distance_l)
        distance_positive_sum_l = np.sum(distance_l[(distance_l>0)&(distance_l<np.inf)])

        #distance to the floor
        row_index = np.tile(np.arange(m_shape[0])+1,(m_shape[1],1)).T
        row_index_base  = np.where(base_matrix_effective!=0,row_index,0)
        row_index_stone = np.where(stone.matrix!=0,row_index,np.inf)
        mask_b = np.zeros(m_shape)
        mask_b[0:int(stone.center[1]),:] = 1
        row_index_base_masked = np.multiply(row_index_base,mask_b)
        base_tb = np.max(row_index_base_masked,axis = 0)
        stone_bb = np.min(row_index_stone,axis = 0)
        distance_b = stone_bb-base_tb-1
        distance_positive_sum_b = np.sum(distance_b[(distance_b>0)&(distance_b<np.inf)])

        #void_ratio = 0.5*distance_positive_sum_l/(m_shape[1]*stone.height)+0.5*distance_positive_sum_b/(m_shape[0]*stone.width)
        void_ratio = (distance_positive_sum_l+distance_positive_sum_b)/(m_shape[1]*stone.height+m_shape[0]*stone.width)

    distance += void_ratio
    # _____________________________________________________________use local void where stone is placed in the center
    local_center = _base_after[max(0, int(stone.center[1]-stone.height/2-local_width_h)):min(stone.matrix.shape[0], int(stone.center[1] + stone.height/2)),
                        max(0, int(stone.center[0]-stone.width/2-local_width_w/2)):min(stone.matrix.shape[1], int(stone.center[0]+stone.width/2+local_width_w/2))]
    nb_void_center = len(np.argwhere(local_center == 0))
    void_ratio_center = nb_void_center/((stone.height+local_width_h)*(stone.width+local_width_w)-stone.height*stone.width)
    # --------------------------------------------------------------------------global density
    occupied = np.argwhere(np.where(
        base.matrix != get_ignore_pixel_value(), base.matrix, 0))
    bbox = np.zeros(base.matrix.shape)
    if occupied.any():
        _max_stone_bb = occupied.max(axis=0)
        _min_stone_bb = occupied.min(axis=0)
        bbox[_min_stone_bb[0]:_max_stone_bb[0]+1,
             _min_stone_bb[1]:base.matrix.shape[1]] = 1
    else:
        bbox[0:3, :] = 1
    distance_gd = np.argwhere(np.multiply(bbox, stone.matrix)
                              ).shape[0]/np.argwhere(bbox+stone.matrix).shape[0]
    distance_gd = distance_gd / \
        (np.argwhere(stone.matrix).shape[0]/np.argwhere(bbox).shape[0])
    if distance_gd > 0.7:
        distance_gd *= 1
    else:
        distance_gd *= 0
    distance -= distance_gd
    # --------------------------------------------------------------------------x distance
    
    x_distance = stone.center[0]/stone.matrix.shape[1]
    #if get_placement_optimization() != "image_convolve": 
    distance += x_distance
    # write to txt
    if write_txt == True:
        with open(get_dir()+f"record/place_ite{len(base.placed_bricks)}_brick{stone.id}_{stone.width}_{stone.rotate_from_ori}.txt", "a+") as f:
            f.write("{};{};{};{};{};{}\n".format(stone.center[0], stone.center[1],
                    distance, void_ratio, -distance_gd,x_distance))
    return distance,-void_ratio,-void_ratio_center


def get_distance_to_interlocking(matrix):
    # revert 0 and 1
    matrix_boundary = np.where(matrix==0,1,0)
   
    # check if there is any zero pixel
    if len(np.argwhere(matrix_boundary==0))==0:
        return np.ones_like(matrix_boundary)*0.12
    # compute the distance to the boundary
    distance_matrix = scipy.ndimage.distance_transform_edt(matrix_boundary, return_distances=True, return_indices=False)
    return distance_matrix

import cv2
import math
def compute_edge(matrix):
    kernel = np.ones((1,3),np.int8)
    #the second half of the kernel is -1
    kernel[:,1] = 0
    kernel[:,2] = -1
    #convolve
    edge = ndimage.convolve(
        matrix, kernel, mode='nearest')
    edge = np.where(edge!=0,1,0)
    return edge

def get_distance_to_edge(matrix):   
    # plt.imshow(abs_sobel64f)
    # plt.show()
    matrix_boundary = compute_edge(matrix)
    # plt.imshow(matrix_boundary)
    # plt.show()
    # erosion and dilation (cv2 closing not working as expected)
    kernel = np.ones((NUMBER_OF_PIXELS_AS_EPSILON,1),np.uint8)
    matrix_boundary = cv2.erode(matrix_boundary.astype('uint8'),kernel,iterations = 1)
    matrix_boundary = cv2.dilate(matrix_boundary.astype('uint8'),kernel,iterations = 1)
    # revert 0 and 1
    matrix_boundary = np.where(matrix_boundary==0,1,0)
   
    # check if there is any zero pixel
    if len(np.argwhere(matrix_boundary==0))==0:
        return np.ones_like(matrix_boundary)*0.12
    # compute the distance to the boundary
    distance_matrix = scipy.ndimage.distance_transform_edt(matrix_boundary, return_distances=True, return_indices=False)
    return distance_matrix


def get_phi_distance(base):
    #ground_pixel_value = get_ignore_pixel_value()
    #base = np.where(base==ground_pixel_value,0,base)
    # compute the minimal distance
    base_reverted  = np.where(base!=0,0,1)
    new_phi = scipy.ndimage.distance_transform_edt(base_reverted, return_distances=True,
                                         return_indices=False)
    return new_phi

def get_height(matrix):
    # matrix where the last row in dimension 0 is 0, the others are 1
    height_matrix = np.ones_like(matrix)
    height_matrix[0,:] = 0
    height_matrix = scipy.ndimage.distance_transform_edt(height_matrix, return_distances=True, return_indices=False)
    # normalize
    height_matrix = height_matrix/np.max(height_matrix)
    return height_matrix

def get_convolution_metric(base_phi, brick_bounding_box_matrix_mirrow):
    shift_to = [(brick_bounding_box_matrix_mirrow.shape[0]-1) // 2,
               (brick_bounding_box_matrix_mirrow.shape[1]-1) // 2]
    proximity_metric = ndimage.convolve(
       base_phi, brick_bounding_box_matrix_mirrow, mode='constant', cval=1.0, origin=shift_to)
    #proximity_metric = np.multiply(base_phi, stone)
    return proximity_metric

def evaluate_convolve(base,brick,brick_bounding_box_matrix_mirrow=None,region_potential=None):
    stone_width = brick.width
    stone_height = brick.height
    stone_center = (brick.width/2,brick.height/2)
    brick_bounding_box_matrix = regionprops(brick.matrix.astype(np.uint8))[0].image
    shift_to = [(brick_bounding_box_matrix_mirrow.shape[0]-1) // 2,(brick_bounding_box_matrix_mirrow.shape[1]-1) // 2]

    # find positions that have support from both left and right
    kernel_dilation = np.ones((3, 3), np.uint8)
    wall_with_no_bound = np.where(base.matrix!=get_ignore_pixel_value(),base.matrix,0)
    contour_dilated = cv2.dilate(wall_with_no_bound, kernel_dilation,
                            anchor=(1, 1), iterations=1)
    
    #dilate the region to filter out floating positions
    # kernel_dilation_y = np.ones((2, 1), np.uint8)
    # contour_dilated = cv2.dilate(base.matrix, kernel_dilation_y, anchor=(0,1),iterations=1)

    # left contact
    left_mask = np.zeros_like(brick_bounding_box_matrix_mirrow)
    # print(brick_bounding_box_matrix_mirrow.shape)
    # print(math.floor(stone_center[0]))
    left_mask[:,0:math.floor(stone_center[0])-2] = 1
    left_brick_bounding_box_matrix_mirrow = brick_bounding_box_matrix_mirrow*left_mask
    left_overlap_contour_mask = np.where(ndimage.convolve(
        contour_dilated, left_brick_bounding_box_matrix_mirrow, mode='constant', cval=0.0, origin=shift_to) != 0, 1, 0)
    #right contact
    right_mask = np.zeros_like(brick_bounding_box_matrix_mirrow)
    right_mask[:,math.ceil(stone_center[0])+2:] = 1
    right_brick_bounding_box_matrix_mirrow = brick_bounding_box_matrix_mirrow*right_mask
    right_overlap_contour_mask = np.where(ndimage.convolve(
        contour_dilated, right_brick_bounding_box_matrix_mirrow, mode='constant', cval=0.0, origin=shift_to) != 0, 1, 0)
    
    stable_potential = np.multiply(right_overlap_contour_mask, left_overlap_contour_mask)
    if len(np.argwhere(np.multiply(region_potential,stable_potential)!=0))==0:
        #print("No stable potential")
        stable_potential = np.ones_like(region_potential)
    #stable_potential = np.ones_like(region_potential)
    region_potential = np.multiply(region_potential,stable_potential)

    # find locations matches mason's practice
    placed_stone_left_tops = np.asarray(base.rock_left_tops)
    placed_stone_right_tops = np.asarray(base.rock_right_tops)
    #interlocking point matrix
    interlocking_point_matrix = np.zeros(base.matrix.shape)
    #make all top points of placed stones 1
    if placed_stone_left_tops.shape[0] >= 1:
        interlocking_point_matrix[placed_stone_left_tops[:, 1], placed_stone_left_tops[:, 0]] = 1
    if placed_stone_right_tops.shape[0] >= 1:
        interlocking_point_matrix[placed_stone_right_tops[:, 1], placed_stone_right_tops[:, 0]] = 1
    # plt.imshow(interlocking_point_matrix)
    # plt.title("interlocking point")
    # plt.show()

    
    
    #left interlocking distance map and convolve with the bounding of stone
    interlocking_distance = get_distance_to_interlocking(interlocking_point_matrix)#maximize the distance to the interlocking
    brick_bounding_box_matrix_boundary_only = np.zeros_like(brick_bounding_box_matrix)#using bounding box instead of actual stone shape
    brick_bounding_box_matrix_boundary_only[0,0] = 1
    left_interlocking_distance_convolved = ndimage.minimum_filter(
        interlocking_distance, footprint=brick_bounding_box_matrix_boundary_only, mode='constant', cval=np.inf, origin=[-shift_to[0],-shift_to[1]])
    left_interlocking_distance_convolved[left_interlocking_distance_convolved==0.12] = np.inf
    #right interlocking distance map and convolve with the bounding of stone
    brick_bounding_box_matrix_boundary_only = np.zeros_like(brick_bounding_box_matrix)#using bounding box instead of actual stone shape
    brick_bounding_box_matrix_boundary_only[0,-1] = 1
    right_interlocking_distance_convolved = ndimage.minimum_filter(
        interlocking_distance, footprint=brick_bounding_box_matrix_boundary_only, mode='constant', cval=np.inf, origin=[-shift_to[0],-shift_to[1]])
    right_interlocking_distance_convolved[right_interlocking_distance_convolved==0.12] = np.inf
    
    #touch

    touch_distance = get_distance_to_edge(base.matrix)
    brick_bounding_box_matrix_boundary_only = np.zeros_like(brick_bounding_box_matrix)#using bounding box instead of actual stone shape
    #print("The stone center is at ",stone_center[0])
    #print("The stone center int is at ",int(stone_center[0]))
    brick_bounding_box_matrix_boundary_only[math.ceil(stone_center[1]):,0] = 1
    left_touch_distance_convolved = ndimage.minimum_filter(
        touch_distance, footprint=brick_bounding_box_matrix_boundary_only, mode='constant', cval=0, origin=[-shift_to[0],-shift_to[1]])
    #right interlocking distance map and convolve with the bounding of stone
    brick_bounding_box_matrix_boundary_only = np.zeros_like(brick_bounding_box_matrix)#using bounding box instead of actual stone shape
    brick_bounding_box_matrix_boundary_only[math.ceil(stone_center[1]):,-1] = 1
    right_touch_distance_convolved = ndimage.minimum_filter(
        touch_distance, footprint=brick_bounding_box_matrix_boundary_only, mode='constant', cval=0, origin=[-shift_to[0],-shift_to[1]])
    

    interlocking_thre = min(INTERLOCKING_PIXEL,stone_width/2)
    interlocking_thre = min(interlocking_thre,stone_height)
   
    #mason_criteria = np.where((left_interlocking_distance_convolved>=interlocking_thre)&(right_interlocking_distance_convolved>=interlocking_thre),1,0)
    mason_criteria = np.where(((left_interlocking_distance_convolved>=interlocking_thre)&(right_touch_distance_convolved<=1))|\
    ((right_interlocking_distance_convolved>=interlocking_thre)&(left_touch_distance_convolved<=1))|\
        ((right_touch_distance_convolved<=1)&(left_touch_distance_convolved<=1)),1,0)
    

    iteration_ = 1
    while len(np.argwhere(np.multiply(region_potential,mason_criteria)!=0))==0 and iteration_<4:

        # mason_criteria = np.where((left_interlocking_distance_convolved>=interlocking_thre/(iteration_+1))\
        #                           &(right_interlocking_distance_convolved>=(interlocking_thre/(iteration_+1))))
        mason_criteria = np.where(((left_interlocking_distance_convolved>=interlocking_thre/(iteration_+1))&(right_touch_distance_convolved<=1))|\
        ((right_interlocking_distance_convolved>=interlocking_thre/(iteration_+1))&(left_touch_distance_convolved<=1))|\
        ((right_touch_distance_convolved<=1)&(left_touch_distance_convolved<=1)),1,0)
        iteration_+=1
    if len(np.argwhere(np.multiply(region_potential,mason_criteria)!=0))==0:
        #print("No interlocking positions")
        mason_criteria = np.ones_like(region_potential)

    # plt.imshow(interlocking_point_matrix)
    # plt.title("mason criterial with interlocking thre = {}".format(interlocking_thre))
    # plt.show()
    weight_height = np.sqrt(stone_width**2+stone_height**2)*1000
    base_phi = get_phi_distance(base.matrix)
    base_height = get_height(base.matrix)
    score_optimization_phi = get_convolution_metric(base_phi, brick_bounding_box_matrix_mirrow)#smaller better
    score_optimization_height = get_convolution_metric(base_height*weight_height, brick_bounding_box_matrix_mirrow)
    distance_optimization = score_optimization_phi+score_optimization_height#smaller better
    score_potential = np.where((region_potential != 0) & (mason_criteria!=0), distance_optimization, float("+inf"))
    best_score = np.min(score_potential)
    #best_loc = np.argmax(score_potential)
    best_loc = np.argwhere(score_potential == best_score)[0]
    # get the proximity metric of the best location
    best_distance_optimization = distance_optimization[best_loc[0],best_loc[1]]
    best_phi = -score_optimization_phi[best_loc[0],best_loc[1]]

    # print("Best location",best_loc)
    # print("Best left interlocking",left_interlocking_distance_convolved[best_loc[0],best_loc[1]])
    # print("Best right interlocking",right_interlocking_distance_convolved[best_loc[0],best_loc[1]])
    # print("Best left touch",left_touch_distance_convolved[best_loc[0],best_loc[1]])
    # print("Best right touch",right_touch_distance_convolved[best_loc[0],best_loc[1]])
    # print("Best height distance",score_optimization_height[best_loc[0],best_loc[1]])
    # print("Best void distance",-best_phi)

    
    
    #return best_loc,best_distance_optimization,best_phi,left_touch_distance_convolved[best_loc[0],best_loc[1]],right_touch_distance_convolved[best_loc[0],best_loc[1]]
    return {"best_loc":best_loc,"best_distance":best_distance_optimization,"best_phi":best_phi,"best_left_interlocking":left_interlocking_distance_convolved[best_loc[0],best_loc[1]],\
            "best_right_interlocking":right_interlocking_distance_convolved[best_loc[0],best_loc[1]],"best_left_touch":left_touch_distance_convolved[best_loc[0],best_loc[1]],\
                "best_right_touch":right_touch_distance_convolved[best_loc[0],best_loc[1]]}
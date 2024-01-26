
import matplotlib.pyplot as plt
from copy import deepcopy

import numpy as np

from ..evaluate.tilting_angle import evaluate_kine
from ..evaluate.ksteps import build_kstones
from ..evaluate.local_void import evaluate,evaluate_convolve,get_void_to_left
from ..utils.constant import get_record_detail,get_record_time,get_local_width,get_partial_enumeration_ratio,get_placement_optimization,get_mu,get_width_weight, get_ignore_pixel_value, get_ksteps, get_selection_weights, get_world_size, get_dir
from ..utils.logger import get_main_logger
from scipy.spatial import KDTree
import math
import time
from ..utils.plot import save_id_plot
from skimage.segmentation import watershed
from skimage.measure import regionprops,label
from scipy import ndimage
import multiprocessing
from pyMetaheuristic.algorithm import adaptive_random_search
#from .fda import flow_direction_algorithm
import random
import os
import cv2
from ..utils.plot import save_matrix_plot,save_matrix_with_mask

def isNaN(num):
    return num != num
def seed_everything(seed=20):
    """"
    Seed everything.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def sigmoid(x):
            return 1 / (1 + math.exp(-x))

def place_one_stone(brick_index, all_stones, all_poses, base, locations=None, replace=False, rec_time=True,region = None):
    """Find a position of one stone

    :param brick_index: Brick index in the list of stone poses
    :type brick_index: int
    :param all_stones: List of stones
    :type all_stones: list
    :param all_poses: List of poses of the stones
    :type all_poses: list
    :param base: Current base
    :type base: Stonepacker2D.base.Base
    :param locations: Possible locations
    :type locations: list
    :param replace: Whether refill the stone storage after placement, defaults to True (no refillment)
    :type replace: bool, optional
    :return: Found translation x, y, score of the stone, kinematics result
    :rtype: bool
    """
    write_txt = get_record_detail()['RECORD_SELECTION_VALUE']
    plot_img_detail = get_record_detail()['RECORD_PLACEMENT_IMG']
    seed_everything()#! It is important to reset seed for each processor. If not, the random numbers are not reproducible as processors' computation order is changed every run.
    brick = all_poses[brick_index]

    #convolution to eliminate the overlap
    if get_placement_optimization() == "image_convolve":
        a = base.matrix
        brick_bounding_box_matrix = regionprops(brick.matrix.astype(np.uint8))[0].image
        brick_bounding_box_matrix_mirrow = np.flip(np.flip(brick_bounding_box_matrix,axis = 0),axis=1)
        shift_to = [(brick_bounding_box_matrix_mirrow.shape[0]-1) // 2,(brick_bounding_box_matrix_mirrow.shape[1]-1) // 2]
        non_overlap_mask = np.where(ndimage.convolve(a, brick_bounding_box_matrix_mirrow, mode='constant', cval=1.0,origin = shift_to)==0,1,0)
        #region_non_overlap = np.multiply(region, non_overlap_mask)
        
        #dilate the region to filter out floating positions
        kernel_dilation_y = np.ones((2, 1), np.uint8)
        contour_up = cv2.dilate(a, kernel_dilation_y, anchor=(0,1),iterations=1)
        overlap_contour_mask = np.where(ndimage.convolve(contour_up, brick_bounding_box_matrix_mirrow, mode='constant', cval=0.0,origin = shift_to)!=0,1,0)
        
        #intersection of two constraints
        region_potential = np.multiply(non_overlap_mask, overlap_contour_mask)
        plot_region_matrix = np.where(non_overlap_mask!=0,1,0)+np.where(overlap_contour_mask!=0,2,0)
        locations = np.argwhere(region_potential!=0)
        if plot_img_detail:
            save_matrix_plot(np.where(a !=0, 1, 0), f"ite{len(base.placed_bricks)}_brick{brick.id}_{brick.width}_{brick.rotate_from_ori}_brick_a.png")
            save_matrix_plot(brick.matrix.astype(np.uint8), f"ite{len(base.placed_bricks)}_brick{brick.id}_{brick.width}_{brick.rotate_from_ori}_brick_bounding_box_matrix.png")
            save_matrix_plot(np.where(non_overlap_mask!=0,1,0), f"ite{len(base.placed_bricks)}_brick{brick.id}_{brick.width}_{brick.rotate_from_ori}_non_overlap_mask.png")
            save_matrix_plot(np.where(overlap_contour_mask!=0,1,0), f"ite{len(base.placed_bricks)}_brick{brick.id}_{brick.width}_{brick.rotate_from_ori}_overlap_contour_mask.png")
            save_matrix_plot(region_potential, f"ite{len(base.placed_bricks)}_brick{brick.id}_{brick.width}_{brick.rotate_from_ori}_region_potential.png")
            save_matrix_plot(plot_region_matrix,f"ite{len(base.placed_bricks)}_brick{brick.id}_{brick.width}_{brick.rotate_from_ori}_region_potential_color.png",color = True)
            save_matrix_with_mask(np.where(a !=0, 1, 0),plot_region_matrix,f"ite{len(base.placed_bricks)}_brick{brick.id}_{brick.width}_{brick.rotate_from_ori}_region_mask.png",alpha = 0.2)
        

    times = np.zeros((1,6))
    if get_record_time():
        start = time.time()
    if write_txt:
        with open(get_dir()+f"record/place_ite{len(base.placed_bricks)}_brick{brick.id}_{brick.width}_{brick.rotate_from_ori}.txt", "w+") as f:
            f.write("x;y;score;void;global density;x distance\n")
    if len(np.argwhere(region_potential!=0))==0:
        get_main_logger().warning(
            "No valid location found for stone {}".format(brick.id))
        return -1, -1, float("+inf"),-1*np.ones(6), 0,times
    
    #convolution-based evaluation
    result_evaluation = evaluate_convolve(base,brick,brick_bounding_box_matrix_mirrow,region_potential)
    #best_loc,best_score_optimization,best_phi,left_touch, right_touch = evaluate_convolve(base,brick,brick_bounding_box_matrix_mirrow,region_potential)
    best_loc = result_evaluation['best_loc']
    best_score_optimization = result_evaluation['best_distance']
    #best_phi = result_evaluation['best_phi']
    left_touch = result_evaluation['best_left_touch']
    right_touch = result_evaluation['best_right_touch']
    min_x = best_loc[1]
    min_y = best_loc[0]
    min_distance = best_score_optimization
    #max_local_void = best_phi
    # # initialization
    # min_x = 0
    # min_y = 0
    # min_distance,max_local_void, max_local_void_center = evaluate(base, brick)

    # def distance_trial_placement(variables_values = [0, 0]):
    #     if isNaN(variables_values[0]) or isNaN(variables_values[1]):
    #         return float("+inf")
    #     if [int(variables_values[0]), int(variables_values[1])] not in locations:
    #         return float("+inf")
    #     try_stone = brick.transformed(
    #         [int(variables_values[0]), int(variables_values[1])])
    #     if try_stone is None:
    #         return float("+inf")
    #     else:
    #         distance,_,_ = evaluate(
    #             base, try_stone)
    #         return distance

    # if get_placement_optimization() == "full_enumeration" or get_placement_optimization() == "image_convolve":
    #     for void_coor in locations:
    #         try_stone = brick.transformed(
    #             [void_coor[1], void_coor[0]])
    #         if try_stone is None:
    #             continue
    #         distance,_local_void,_local_void_center = evaluate(
    #             base, try_stone)

    #         if distance < min_distance:
    #             min_distance = distance
    #             max_local_void = _local_void
    #             max_local_void_center = _local_void_center
    #             min_x = void_coor[1]
    #             min_y = void_coor[0]
    #     if get_record_time():
    #         end = time.time()
    #         times[0,0] = len(base.placed_bricks)
    #         times[0,1] = len(locations)
    #         times[0,2] = end-start
    #         # # write to txt
    #         # with open(get_dir()+str(multiprocessing.current_process()._identity)+"_time.txt", "a+") as f:
    #         #     f.write("{};{};{};".format(
    #         #         len(base.placed_bricks), len(locations), end-start))
    # elif get_placement_optimization() == "partial_enumeration":
    #     # get the total number of possible locations and sample randomly
    #     total_location_nb = locations.shape[0]
    #     partial_location_nb = int(total_location_nb*get_partial_enumeration_ratio())
    #     if partial_location_nb == 0:
    #         partial_locations = locations
    #     else:
    #         random_indices = np.random.choice(total_location_nb, partial_location_nb, replace=False)
    #         partial_locations = locations[random_indices,:]
        
    #     # iteration all locations and find the best
    #     for void_coor in partial_locations:
    #         try_stone = brick.transformed(
    #             [void_coor[1], void_coor[0]])
    #         if try_stone is None:
    #             continue
    #         distance,_local_void,_local_void_center = evaluate(
    #             base, try_stone)

    #         if distance < min_distance:
    #             min_distance = distance
    #             max_local_void = _local_void
    #             max_local_void_center = _local_void_center
    #             min_x = void_coor[1]
    #             min_y = void_coor[0]
    #     if get_record_time():
    #         end = time.time()
    #         times[0,0] = len(base.placed_bricks)
    #         times[0,1] = len(partial_locations)
    #         times[0,2] = end-start
    # elif get_placement_optimization() == "ARS":
    #     # Adaptive Random Search algo
    #     # https://github.com/Valdecy/pyMetaheuristic/tree/7eac98316f33a1c401e2b5e204aa25fe599a3fb2
    #     # target function
    #     # variables_values = (x,y)
        
    #     # ARS - Parameters
    #     x_min_bound = np.min(locations,axis = 0)[1]
    #     x_max_bound = np.max(locations,axis = 0)[1]
    #     y_min_bound = np.min(locations,axis = 0)[0]
    #     y_max_bound = np.max(locations,axis = 0)[0]
    #     parameters = {
    #         'solutions': 10,#number of sample per iteration
    #         'min_values': (x_min_bound, y_min_bound),
    #         'max_values': (x_max_bound, y_max_bound),
    #         'step_size_factor': 0.2,
    #         'factor_1': 3,
    #         'factor_2': 1.5,
    #         'iterations': 100,
    #         'large_step_threshold': 15,
    #         'improvement_threshold': 25,#number of iterations with one step size value
    #         'verbose': False
    #     }
    #     ars = adaptive_random_search(target_function = distance_trial_placement, **parameters)
    #     min_x = ars[0][0]
    #     min_y = ars[0][1]
    #     min_distance = ars[0][2]
    #     if get_record_time():
    #         end = time.time()
    #         times[0,0] = len(base.placed_bricks)
    #         times[0,1] = len(locations)
    #         times[0,2] = end-start
    # elif get_placement_optimization() == "FDA":
    #     # Flow Direction Algorithm algo
    #     # https://github.com/Valdecy/pyMetaheuristic/tree/7eac98316f33a1c401e2b5e204aa25fe599a3fb2
    #     # target function
    #     # variables_values = (x,y)

    #     # FDA - Parameters
    #     x_min_bound = np.min(locations,axis = 0)[1]
    #     x_max_bound = np.max(locations,axis = 0)[1]
    #     y_min_bound = np.min(locations,axis = 0)[0]
    #     y_max_bound = np.max(locations,axis = 0)[0]
    #     parameters = {
    #         'size': 10,
    #         'min_values': (x_min_bound, y_min_bound),
    #         'max_values': (x_max_bound, y_max_bound),
    #         'iterations': 100,
    #         'beta': 8,#number of neighbors
    #         'verbose': True
    #     }
    #     fda = flow_direction_algorithm(target_function = distance_trial_placement, **parameters)
    #     min_x = int(fda[0])
    #     min_y = int(fda[1])
    #     min_distance = fda[2]
    #     if get_record_time():
    #         end = time.time()
    #         times[0,0] = len(base.placed_bricks)
    #         times[0,1] = len(locations)
    #         times[0,2] = end-start
    # elif get_placement_optimization() == "partial_enumeration_FDA":
    #     # get the total number of possible locations and sample randomly
    #     total_location_nb = locations.shape[0]
    #     partial_location_nb = int(total_location_nb*get_partial_enumeration_ratio())
    #     if partial_location_nb == 0:
    #         partial_locations = locations
    #     else:
    #         random_indices = np.random.choice(total_location_nb, partial_location_nb, replace=False)
    #         partial_locations = locations[random_indices,:]
        
    #     # iteration all locations and find the best
    #     # local_width = get_local_width()
    #     # if local_width == 0:
    #     #     local_width_w = stone.height
    #     #     local_width_h = stone.width
    #     # else:
    #     #     local_width_w = local_width
    #     #     local_width_h = local_width
    #     for void_coor in partial_locations:
    #         try_stone = brick.transformed(
    #             [void_coor[1], void_coor[0]])
    #         if try_stone is None:
    #             continue
    #         distance,_local_void,_local_void_center = evaluate(
    #             base, try_stone)
            
    #         # if distance <float("+inf"):
    #         #     # FDA - Parameters
    #         #     x_min_bound = -local_width_w/2+void_coor[1]
    #         #     x_max_bound = local_width_w/2+void_coor[1]
    #         #     y_min_bound = -local_width_h/2+ void_coor[0]
    #         #     y_max_bound =  local_width_h/2+void_coor[0]

    #         #     total_size = local_width_w*local_width_h
                
    #         #     sample_ratio = 0.5
    #         #     _size = int(sample_ratio*total_size/(3*5))
    #         #     parameters = {
    #         #         'size': _size,
    #         #         'min_values': (x_min_bound, y_min_bound),
    #         #         'max_values': (x_max_bound, y_max_bound),
    #         #         'iterations': 3,
    #         #         'beta': 4,#number of neighbors
    #         #         'verbose': False
    #         #     }
    #         #     fda = flow_direction_algorithm(target_function = distance_trial_placement, **parameters)
    #         #     min_x_fda = int(fda[0])
    #         #     min_y_fda = int(fda[1])
    #         #     min_distance_fda = fda[2]
    #         #     if min_distance_fda<distance:
    #         #         void_coor[1] = min_x_fda
    #         #         void_coor[0] = min_y_fda
    #         #         distance = min_distance_fda

    #         if distance < min_distance:
    #             min_distance = distance
    #             max_local_void = _local_void
    #             max_local_void_center = _local_void_center
    #             min_x = void_coor[1]
    #             min_y = void_coor[0]
    #     # Flow Direction Algorithm algo
    #     # https://github.com/Valdecy/pyMetaheuristic/tree/7eac98316f33a1c401e2b5e204aa25fe599a3fb2
    #     # target function
    #     # variables_values = (x,y)
    #     local_width = get_local_width()
    #     if local_width == 0:
    #         local_width_w = stone.height
    #         local_width_h = stone.width
    #     else:
    #         local_width_w = local_width
    #         local_width_h = local_width
    #     # FDA - Parameters
    #     x_min_bound = -1/get_partial_enumeration_ratio()+min_x
    #     x_max_bound = 1/get_partial_enumeration_ratio()+min_x
    #     y_min_bound = -1/get_partial_enumeration_ratio()+min_y
    #     y_max_bound = 1/get_partial_enumeration_ratio()+min_y

    #     total_size = (x_max_bound-x_min_bound)*(y_max_bound-y_min_bound)
    #     sample_ratio = 0.4
    #     _size = int(sample_ratio*total_size/(3*9))

    #     parameters = {
    #         'size': _size,
    #         'min_values': (x_min_bound, y_min_bound),
    #         'max_values': (x_max_bound, y_max_bound),
    #         'iterations': 3,
    #         'beta': 8,#number of neighbors
    #         'verbose': False
    #     }
    #     fda = flow_direction_algorithm(target_function = distance_trial_placement, **parameters)
    #     min_x_fda = int(fda[0])
    #     min_y_fda = int(fda[1])
    #     min_distance_fda = fda[2]
    #     if min_distance_fda<min_distance:
    #         min_x = min_x_fda
    #         min_y = min_y_fda
    #         min_distance = min_distance_fda
        
    #     if get_record_time():
    #         end = time.time()
    #         times[0,0] = len(base.placed_bricks)
    #         times[0,1] = len(partial_locations)
    #         times[0,2] = end-start
    # elif get_placement_optimization() == "partial_enumeration_full":
    #     # get the total number of possible locations and sample randomly
    #     total_location_nb = locations.shape[0]
    #     partial_location_nb = int(total_location_nb*get_partial_enumeration_ratio())
    #     if partial_location_nb == 0:
    #         partial_locations = locations
    #     else:
    #         random_indices = np.random.choice(total_location_nb, partial_location_nb, replace=False)
    #         partial_locations = locations[random_indices,:]
        
    #     for void_coor in partial_locations:
    #         try_stone = brick.transformed(
    #             [void_coor[1], void_coor[0]])
    #         if try_stone is None:
    #             continue
    #         distance,_local_void,_local_void_center = evaluate(
    #             base, try_stone)
            

    #         if distance < min_distance:
    #             min_distance = distance
    #             max_local_void = _local_void
    #             max_local_void_center = _local_void_center
    #             min_x = void_coor[1]
    #             min_y = void_coor[0]

    #     x_min_bound = int(-1/get_partial_enumeration_ratio()+min_x)
    #     x_max_bound = int(1/get_partial_enumeration_ratio()+min_x)
    #     y_min_bound = int(-1/get_partial_enumeration_ratio()+min_y)
    #     y_max_bound = int(1/get_partial_enumeration_ratio()+min_y)

    #     for second_stage_x in range(x_min_bound,x_max_bound):
    #         for second_stage_y in range(y_min_bound,y_max_bound):
    #             try_stone = brick.transformed(
    #                 [second_stage_x, second_stage_y])
    #             if try_stone is None:
    #                 continue
    #             distance,_local_void,_local_void_center = evaluate(
    #                 base, try_stone)
    #             if distance < min_distance:
    #                 min_distance = distance
    #                 max_local_void = _local_void
    #                 max_local_void_center = _local_void_center
    #                 min_x = second_stage_x
    #                 min_y = second_stage_y

        
        
    #     if get_record_time():
    #         end = time.time()
    #         times[0,0] = len(base.placed_bricks)
    #         times[0,1] = len(partial_locations)
    #         times[0,2] = end-start
    # # reevaluate stone based on height
    if min_distance == float("+inf"):
        get_main_logger().warning(
            "No valid location found for stone {}".format(brick.id))
        # if rec_time:
        #     with open(get_dir()+str(multiprocessing.current_process()._identity)+"_time.txt", "a") as f:
        #         f.write("\n")
        return min_x, min_y, float("+inf"),-1*np.ones(6), 0,times
    try_stone = brick.transformed(
        [min_x, min_y])
    if get_placement_optimization() != "full_enumeration":
        _,max_local_void,max_local_void_center = evaluate(
                    base, try_stone)
    if get_record_time():
        start = time.time()
    
    # ____________________________________collect info of the data set
    heights_stones = np.zeros(len(all_stones))
    widths_stones = np.zeros(len(all_poses))
    #min_stone_area = np.inf
    for s, stone in enumerate(all_stones):
        heights_stones[s] = stone.height
    for s,stone in enumerate(all_poses):
        widths_stones[s] = stone.width
        # if stone.area < min_stone_area:
        #     min_stone_area = stone.area

    wall_width = base.matrix.shape[1]
    selection_weights = get_selection_weights()
    filters = np.zeros(6)
    #filters[4] = max_local_void
    filters[4] = -get_void_to_left(base,try_stone)
    filters[5] = max_local_void_center
    #_____________________________________place and stabalize
    # build k stones ahead
    future_base = deepcopy(base)
    # add current brick to future base
    future_base.add_stone(brick, [min_x, min_y])
    try_stone = future_base.placed_rocks[-1]

    # ___________________________find the height of left stone based on left top point

    placed_stone_right_tops = np.asarray(base.rock_right_tops)
    if placed_stone_right_tops.shape[0] >= 1:
        placed_stone_right_tops_left = placed_stone_right_tops[placed_stone_right_tops[:, 0]
                                                               < try_stone.center[0]]
    else:
        placed_stone_right_tops_left = []

    stone_h = np.max(np.nonzero(try_stone.matrix)[0])
    if len(placed_stone_right_tops_left) >= 1:
        tree = KDTree(placed_stone_right_tops_left)
        query_point = [np.min(np.nonzero(try_stone.matrix)
                              [1]), np.max(np.nonzero(try_stone.matrix)
                                           [0])]  # min_col, max_row => left top
        _, ii = tree.query(query_point, k=1)
        left_stone_h = placed_stone_right_tops_left[ii][1]
        #min_distance = selection_weights['course_height'] * \
        #    (abs(stone_h-left_stone_h)/try_stone.height)
        filters[0] = -abs(stone_h-left_stone_h)/try_stone.height
    else:
        # no stone is placed on the left of the current stone
        # find the height of the base
        # base_h = np.max(np.nonzero(base.matrix)[0])
        # future_base_h = np.max(np.nonzero(future_base.matrix)[0])
        # if base_h>=future_base_h:#adding the current stone does not increase the height of the base
        #     filters[0] = -abs(base_h-stone_h)/base_h
        # else:

        #     #min_distance = selection_weights['course_height'] * \
        #     #    (abs(np.percentile(heights_stones, 50)-try_stone.height)/try_stone.height)
        filters[0] = -abs(np.percentile(heights_stones, 50) -
                        try_stone.height)/try_stone.height
            # print(np.median(heights_stones))
        #filters[0] = -1
    
    #rec1 = []
    #rec1.append(min_distance)
    # --------------------------------------------------------------------global density
    occupied = np.argwhere(np.where(
        base.matrix != get_ignore_pixel_value(), base.matrix, 0))
    bbox = np.zeros(base.matrix.shape)
    _heighest_stone = 0
    if occupied.any():
        _max_stone_bb = occupied.max(axis=0)
        _min_stone_bb = occupied.min(axis=0)
        bbox[_min_stone_bb[0]:_max_stone_bb[0]+1,
             _min_stone_bb[1]:base.matrix.shape[1]] = 1  # !different bounding box
        _heighest_stone = _max_stone_bb[0]
    else:
        bbox[0:3, :] = 1
        _heighest_stone = 3
    distance_gd = np.argwhere(np.multiply(bbox, try_stone.matrix)
                              ).shape[0]/np.argwhere(bbox+try_stone.matrix).shape[0]
    distance_gd = distance_gd / \
        (np.argwhere(try_stone.matrix).shape[0]/np.argwhere(bbox).shape[0])
    
    if distance_gd > 0.7:
        filters[1] = distance_gd
    else:
        filters[1] = 0
    # min_distance -= distance_gd
    # rec1.append(-distance_gd)
# _______________________________________________________________________interlocking based on right bottom
    #determine if the stone is in the middle or at the edge
    query_point = [np.max(np.nonzero(try_stone.matrix)[1]), np.min(
        np.nonzero(try_stone.matrix)[0])]  # max col, min row
    query_point_2 = [np.min(np.nonzero(try_stone.matrix)[1]), np.min(
        np.nonzero(try_stone.matrix)[0])]  # min col, min row
    distance_to_right_wall_bound = wall_width - query_point[0]
    distance_to_right_wall_bound_2 = query_point_2[0]

    placed_stone_left_tops = np.asarray(base.rock_left_tops)
    placed_stone_right_tops = np.asarray(base.rock_right_tops)
    placed_stone_centers = np.asarray(base.rock_centers)
    stone_b = np.min(np.nonzero(try_stone.matrix)[0])
    if placed_stone_left_tops.shape[0] > 0:
        placed_stone_left_tops_under = placed_stone_left_tops[placed_stone_centers[:, 1] < stone_b]
        placed_stone_right_tops_under = placed_stone_right_tops[placed_stone_centers[:, 1] < stone_b]
    else:
        placed_stone_left_tops_under = []
        placed_stone_right_tops_under = []
    if len(placed_stone_left_tops_under) >= 1:
        # the left edge
        tree = KDTree(placed_stone_left_tops_under)
        distance_ii, ii = tree.query(query_point, k=1)
        _, ii_2 = tree.query(query_point_2, k=1)
        # the right edge
        tree_right = KDTree(placed_stone_right_tops_under)
        _, ii_right = tree_right.query(query_point, k=1)
        distance_ii_right_2, ii_right_2 = tree_right.query(query_point_2, k=1)
        #query point on the right bottom corner of the current stone
        # horizontal_dist_1 = abs(query_point[0]-placed_stone_left_tops_under[ii][0])/(
        #     placed_stone_right_tops_under[ii][0]-placed_stone_left_tops_under[ii][0])
        horizontal_dist_1 = abs(distance_ii)/(
            placed_stone_right_tops_under[ii][0]-placed_stone_left_tops_under[ii][0])
        # horizontal_dist_2 = abs(query_point[0]-placed_stone_right_tops_under[ii_right][0])/(
        #     placed_stone_right_tops_under[ii_right][0]-placed_stone_left_tops_under[ii_right][0])
        # horizontal_dist = min(horizontal_dist_1, horizontal_dist_2)
        horizontal_dist = horizontal_dist_1
        #print("Query point",query_point)
        #print("left nearest point",placed_stone_left_tops_under[ii])
        #print("right nearest point",placed_stone_right_tops_under[ii_right])
        #print("horizontal_dist",horizontal_dist)

        #query point on the left bottom corner of the current stone
        # horizontal_dist_1_2 = abs(query_point_2[0]-placed_stone_left_tops_under[ii_2][0])/(
        #     placed_stone_right_tops_under[ii_2][0]-placed_stone_left_tops_under[ii_2][0])
        # horizontal_dist_2_2 = abs(query_point_2[0]-placed_stone_right_tops_under[ii_right_2][0])/(
        #     placed_stone_right_tops_under[ii_right_2][0]-placed_stone_left_tops_under[ii_right_2][0])
        horizontal_dist_2_2 = abs(distance_ii_right_2)/(
            placed_stone_right_tops_under[ii_right_2][0]-placed_stone_left_tops_under[ii_right_2][0])
        # horizontal_dist_2 = min(horizontal_dist_1_2, horizontal_dist_2_2)
        horizontal_dist_2 = horizontal_dist_2_2
        #print("Query point",query_point_2)
        #print("left nearest point",placed_stone_left_tops_under[ii_2])
        #print("right nearest point",placed_stone_right_tops_under[ii_right_2])
        #print("horizontal_dist_2",horizontal_dist_2)

    else:
        # first layer when no stones under
        horizontal_dist = 0.5
        horizontal_dist_2 = 0.5

    # if distance_to_right_wall_bound < widths_stones.min():
    #     #it is an edge stone on the right
    #     right_mask = np.zeros(try_stone.matrix.shape)
    #     right_mask[:,math.floor(try_stone.center[0]):] = 1
    #     stone_right_matrix = np.multiply(try_stone.matrix, right_mask)
    #     p_top_right = [np.max(np.nonzero(stone_right_matrix)[1]), np.max(
    #     np.nonzero(stone_right_matrix)[0])]# max col, max row
    #     left_mask = np.zeros(try_stone.matrix.shape)
    #     left_mask[:,0:max(1,math.ceil(try_stone.center[0]))] = 1
    #     stone_left_matrix = np.multiply(try_stone.matrix, left_mask)
    #     p_top_left = [np.min(np.nonzero(stone_left_matrix)[1]), np.max(
    #     np.nonzero(stone_left_matrix)[0])]  # min col, max row ---x,y

    #     if p_top_right[1]>p_top_left[1]+2:
    #         incline_tangent = sigmoid((p_top_right[1]-p_top_left[1])/(p_top_right[0]-p_top_left[0]))
    #         #incline_tangent = 0
    #     else:
    #         incline_tangent = horizontal_dist_2
    #     filters[2] = 0.5*(incline_tangent+horizontal_dist_2)

    #     # #change the local void metric to the right side
    #     # m_shape = base.matrix.shape
    #     # #distance to the left
    #     # column_index = np.tile(np.arange(m_shape[1])+1,(m_shape[0],1))
    #     # column_index_stone = np.where(try_stone.matrix!=0,column_index,0)
    #     # stone_rb = np.max(column_index_stone,axis = 1)
    #     # distance_r = m_shape[1]-stone_rb
    #     # # print("base_rb",base_rb)
    #     # # print("stone lb",stone_lb)
    #     # # print("distance",distance_l)
    #     # distance_positive_sum_r = np.sum(distance_r[(distance_r>0)&(distance_r<(m_shape[1]-1))])
    #     # filters[4] = -(distance_positive_sum_r/(m_shape[1]*try_stone.height))
        
    # elif distance_to_right_wall_bound_2 < widths_stones.min():
    #     #edge stone on the left
    #     right_mask = np.zeros(try_stone.matrix.shape)
    #     right_mask[:,math.floor(try_stone.center[0]):] = 1
    #     stone_right_matrix = np.multiply(try_stone.matrix, right_mask)
    #     p_top_right = [np.max(np.nonzero(stone_right_matrix)[1]), np.max(
    #     np.nonzero(stone_right_matrix)[0])]# max col, max row
    #     left_mask = np.zeros(try_stone.matrix.shape)
    #     left_mask[:,0:max(1,math.ceil(try_stone.center[0]))] = 1
    #     stone_left_matrix = np.multiply(try_stone.matrix, left_mask)
    #     p_top_left = [np.min(np.nonzero(stone_left_matrix)[1]), np.max(
    #     np.nonzero(stone_left_matrix)[0])]  # min col, max row ---x,y

    #     if p_top_right[1]>p_top_left[1]+2:
    #         incline_tangent = sigmoid((p_top_left[1]-p_top_right[1])/(p_top_right[0]-p_top_left[0]))
    #         #incline_tangent = 0
    #     else:
    #         incline_tangent = horizontal_dist
        
    #     #find the interlocking of right foot
    #     filters[2] = 0.5*(incline_tangent+horizontal_dist)
        

    #     # #change the local void metric to the right side
    #     # m_shape = base.matrix.shape
    #     # #distance to the left
    #     # column_index = np.tile(np.arange(m_shape[1])+1,(m_shape[0],1))
    #     # column_index_stone = np.where(try_stone.matrix!=0,column_index,np.inf)
    #     # stone_lb = np.min(column_index_stone,axis = 1)
    #     # distance_l = stone_lb-1
    #     # # print("stone lb",stone_lb)
    #     # # print("distance",distance_l[(distance_l>0)&(distance_l<(np.inf-1))])
    #     # distance_positive_sum_l = np.sum(distance_l[(distance_l>0)&(distance_l<(np.inf-1))])
    #     # filters[4] = -(distance_positive_sum_l/(m_shape[1]*try_stone.height))
    #     # #get_main_logger().warning(
    #     # #    f"Stone {try_stone.id} is an edge stone, with tangent {incline_tangent}, metric 2 {filters[2]}, metric 4 {filters[4]}")
    # else:
        
    #     filters[2] = (horizontal_dist+horizontal_dist_2)/2

    if left_touch<=1 and right_touch<=1:
        filters[2] = 0.5
    elif left_touch<=1:
        filters[2] = horizontal_dist
    elif right_touch<=1:
        filters[2] = horizontal_dist_2
    else:
        filters[2] = (horizontal_dist+horizontal_dist_2)/2
    #print("left Horizontal_dist",horizontal_dist)
    #print("right Horizontal_dist",horizontal_dist_2)

    # # print(placed_stone_centers.shape)
    # placed_stone_left_tops = np.asarray(base.rock_left_tops)
    # placed_stone_right_tops = np.asarray(base.rock_right_tops)
    # placed_stone_centers = np.asarray(base.rock_centers)
    # stone_b = np.min(np.nonzero(try_stone.matrix)[0])
    # if placed_stone_left_tops.shape[0] > 0:
    #     placed_stone_left_tops_under = placed_stone_left_tops[placed_stone_centers[:, 1] < stone_b]
    #     placed_stone_right_tops_under = placed_stone_right_tops[placed_stone_centers[:, 1] < stone_b]
    # else:
    #     placed_stone_left_tops_under = []
    #     placed_stone_right_tops_under = []
    # query_point = [np.max(np.nonzero(try_stone.matrix)[1]), np.min(
    #     np.nonzero(try_stone.matrix)[0])]  # max col, min row
    # query_point_2 = [np.min(np.nonzero(try_stone.matrix)[1]), np.min(
    #     np.nonzero(try_stone.matrix)[0])]  # min col, min row
    # if len(placed_stone_left_tops_under) >= 1:
    #     # the left edge
    #     tree = KDTree(placed_stone_left_tops_under)
    #     _, ii = tree.query(query_point, k=1)
    #     _, ii_2 = tree.query(query_point_2, k=1)
    #     # the right edge
    #     tree_right = KDTree(placed_stone_right_tops_under)
    #     _, ii_right = tree_right.query(query_point, k=1)
    #     _, ii_right_2 = tree_right.query(query_point_2, k=1)

    #     distance_to_right_wall_bound = wall_width - query_point[0]

    #     if distance_to_right_wall_bound < widths_stones.min():
    #         horizontal_dist = 1-distance_to_right_wall_bound/widths_stones.min()
    #     else:
    #         horizontal_dist_1 = abs(query_point[0]-placed_stone_left_tops_under[ii][0])/(
    #             placed_stone_right_tops_under[ii][0]-placed_stone_left_tops_under[ii][0])
    #         horizontal_dist_2 = abs(query_point[0]-placed_stone_right_tops_under[ii_right][0])/(
    #             placed_stone_right_tops_under[ii_right][0]-placed_stone_left_tops_under[ii_right][0])
    #         horizontal_dist = min(horizontal_dist_1, horizontal_dist_2)

    #     distance_to_right_wall_bound_2 = query_point_2[0]

    #     if distance_to_right_wall_bound_2 < widths_stones.min():
    #         horizontal_dist_2 = 1-distance_to_right_wall_bound_2/widths_stones.min()
    #     else:
    #         horizontal_dist_1_2 = abs(query_point_2[0]-placed_stone_left_tops_under[ii_2][0])/(
    #             placed_stone_right_tops_under[ii_2][0]-placed_stone_left_tops_under[ii_2][0])
    #         horizontal_dist_2_2 = abs(query_point_2[0]-placed_stone_right_tops_under[ii_right_2][0])/(
    #             placed_stone_right_tops_under[ii_right_2][0]-placed_stone_left_tops_under[ii_right_2][0])
    #         horizontal_dist_2 = min(horizontal_dist_1_2, horizontal_dist_2_2)
    #     # min_distance -= selection_weights['interlocking'] * \
    #     #     (horizontal_dist+horizontal_dist_2)/2

    #     # # min_distance += 1*(horizontal_dist)
    #     # rec1.append(-selection_weights['interlocking']
    #     #             * (horizontal_dist+horizontal_dist_2)/2)
    #     filters[2] = (horizontal_dist+horizontal_dist_2)/2
    # else:  # first layer when no stones under
    #     distance_to_right_wall_bound = wall_width - query_point[0]
    #     if distance_to_right_wall_bound < widths_stones.min():
    #         horizontal_dist = 1-distance_to_right_wall_bound/widths_stones.min()
    #     else:
    #         horizontal_dist = 0.5
    #     # min_distance -= selection_weights['interlocking']*horizontal_dist
    #     # rec1.append(-selection_weights['interlocking']*horizontal_dist)
    #     filters[2] = horizontal_dist

    if get_record_time():
        end = time.time()
        times[0,3] = end-start
        # # write to txt
        # with open(get_dir()+str(multiprocessing.current_process()._identity)+"_time.txt", 'a') as f:
        #     f.write("{};".format(end - start))
        #print("Time for typology evaluation: ", end - start)
        start = time.time()

    
    if get_record_time():
        end = time.time()
        times[0,4] = end-start
        # # write to txt
        # with open(get_dir()+str(multiprocessing.current_process()._identity)+"_time.txt", 'a') as f:
        #     f.write("{};".format(end - start))
        #print("Time for adding brick to future base: ", end - start)
        start = time.time()
    #_______________________________________________________________________________________hole detection
    # # check if there is a hole in the image
    # img = future_base.matrix.astype(np.uint8)
    # markers = np.zeros_like(img)
    # markers[img == 0] = 1
    # #segmentation = watershed(img, markers)
    # labeled_img = label(markers,background=0)
    # regions = regionprops(labeled_img)
    # hole_threshold = min_stone_area
    # #hole_radius_threshold = np.sqrt(hole_threshold/np.pi)
    # for region in regions:
    #     if region.area>=hole_threshold and region.bbox[2]<=_heighest_stone:
    #         return min_x, min_y, -1*np.ones(5), 0


    #_______________________________________________________________________________________end hole detection
    # future_base.plot()
    kine = evaluate_kine(future_base,load = 'tilting_table')/get_mu()
    if get_record_time():
        end = time.time()
        times[0,5] = end-start
        # # write to txt
        # with open(get_dir()+str(multiprocessing.current_process()._identity)+"_time.txt", 'a') as f:
        #     f.write("{}\n".format(end - start))
        #print("Time for kine evaluation: ", end - start)
        start = time.time()
    filters[3] = kine
    # if kine <= 0 and selection_weights['kinematic'] != 0:
    #     min_distance = float("+inf")
    # elif kine <= 0 and selection_weights['kinematic'] == 0:
    #     pass
    # else:
    #     min_distance += selection_weights['kinematic']/kine

    # if get_ksteps() > 0:
    #     # remove current brick from future available stones
    #     future_bricks = deepcopy(all_stones)
    #     if not replace:
    #         _nb_bricks = len(all_stones)
    #         if brick_index < _nb_bricks:
    #             future_bricks.pop(brick_index)
    #         elif brick_index < 2*_nb_bricks:
    #             _ = future_bricks.pop(brick_index-_nb_bricks)
    #         elif brick_index < 3*_nb_bricks:
    #             _ = future_bricks.pop(brick_index-2*_nb_bricks)
    #         # else:
    #         #     _nb_bricks = len(current_bricks)
    #         #     _ = future_bricks.pop(brick_index-3*_nb_bricks)

    #     future_distance, future_base = build_kstones(
    #         future_base, future_bricks, replace=replace)
    #     if future_distance == 0:
    #         future_distance = min_distance
    # else:
    #     future_distance = min_distance

    # write to txt
    if write_txt:
        with open(get_dir()+'img/selection.txt', 'a+') as f:
            f.write("{};{};{};{};{};{};{};{};{};{};{}\n".format(len(base.placed_bricks),
                                                            brick.id, brick.width, brick.center[0]+min_x, brick.center[1]+min_y, min_distance, filters[0], filters[1], filters[2],filters[4], filters[3]))
        
    #min_distance = future_distance
    # if try_stone.width < try_stone.height:
    #     min_distance *= 1.5
    return min_x, min_y, min_distance,filters, kine,times


import matplotlib.pyplot as plt
import cv2
from copy import deepcopy

import numpy as np

from .local_void import evaluate
from .tilting_angle import evaluate_kine
from ..utils.constant import get_ignore_pixel_value, get_rotate_state, get_dilation, get_erosion, get_ksteps, get_width_weight, get_interlocking_weight

from ..utils.logger import get_main_logger
from scipy.spatial import KDTree


def build_kstones(base, bricks, replace=True):
    """Build k stones greedily from current base and current available stones.
    Each step, take the stone that maximize the kinematics limit force

    :param base: Current base
    :type base: Stonepacker2D.base.Base
    :param bricks: Current available stones
    :type bricks: list of Stonepacker2D.stone.Stone
    :param replace: Whether to refill the stone storage after placing
    :type replace: bool, optional; True means refill, False means not refill
    :return: The maximal kinematics limit force after k stones being placed. The base with
    :rtype: float
    """
    # check if there's stones available
    if len(bricks) == 0:
        return 0, base
    base = deepcopy(base)
    current_bricks = deepcopy(bricks)

    # optimization solve
    rotated_bricks_90 = []
    rotated_bricks_180 = []
    # rotated_bricks_270 = []
    if get_rotate_state():
        for i, b in enumerate(current_bricks):
            rotate_90 = b.rotated_by_90()
            rotated_bricks_90.append(rotate_90)
            rotate_180 = rotate_90.rotated_by_90()
            rotated_bricks_180.append(rotate_180)
            # rotate_270 = rotate_180.rotated_by_90()
            # rotated_bricks_270.append(rotate_270)
    i = 1
    while True:
        brick = current_bricks[0]
        min_distance = evaluate(
            base, brick)
        # loop to choose one placement
        min_x = 0
        min_y = 0
        min_brick = brick
        min_index = 0

        occupied = np.argwhere(base.matrix)  # !return index
        if occupied.any():
            # find bounding box
            _max_stone_bb = occupied.max(axis=0)
            _min_stone_bb = occupied.min(axis=0)
            bbox = np.zeros(base.matrix.shape)
            bbox[_min_stone_bb[0]:_max_stone_bb[0]+2,
                 _min_stone_bb[1]:_max_stone_bb[1]+2] = 1

            # oriented contour
            img = np.expand_dims(base.matrix, axis=2)
            img_reverted = img*-1
            kernel_dilation_y = np.ones((get_dilation(), 1), np.uint8)
            kernel_erosion_y = np.ones((1, get_erosion()), np.uint8)
            kernel_dilation_x = np.ones((1, get_dilation()), np.uint8)
            kernel_erosion_x = np.ones(
                (get_erosion(), 1), np.uint8)
            # horizontal contours:detect-expand-erode noise
            sobely = cv2.Sobel(img_reverted, cv2.CV_8U, 0, 1, ksize=-1)
            img_dilation_y = cv2.dilate(
                sobely, kernel_dilation_y, iterations=1)
            img_erosion_y = cv2.erode(
                img_dilation_y, kernel_erosion_y, iterations=1)
            # vertical contour:detect-expand-erode noise
            sobelx = cv2.Sobel(img_reverted, cv2.CV_8U, 1, 0, ksize=-1)
            img_dilation_x = cv2.dilate(
                sobelx, kernel_dilation_x, iterations=1)
            img_erosion_x = cv2.erode(
                img_dilation_x, kernel_erosion_x, iterations=1)
            # bottom line
            floor = np.zeros(base.matrix.shape)
            floor[0, :] = 1
            # concatenate
            ocontour = img_erosion_x+img_erosion_y+floor

            # void
            void = np.where(base.matrix == 0, 1, 0)

            # intersection of the above
            region = np.multiply(bbox, ocontour)
            region = np.multiply(region, void)
            # plt.imshow(region)
            # plt.show()
            locations = np.argwhere(region)
        else:  # no stone placed
            locations = [[0, 0]]

        # iterate over bricks:
        min_distance = float("+inf")
        for brick_index, brick in enumerate(current_bricks+rotated_bricks_90+rotated_bricks_180):
            # brick.plot()
            # initialize min_distance for each brick

            x, y, distance = _place_once(
                brick_index, current_bricks, current_bricks+rotated_bricks_90+rotated_bricks_180, base, locations)

            if distance < min_distance:
                min_distance = distance
                min_brick = brick
                min_x = x
                min_y = y
        if min_distance == float("+inf"):
            get_main_logger().warning(
                f"Cannot find a placement for any {len(current_bricks)} stones given, when forcasting {i} step")
            break

        brick = min_brick
        # ! make sure that the brick is the one from solution
        base.add_stone(brick, [min_x, min_y])
        # print(f"brick added to base: {brick.width} * {brick.height}")
        if not replace:
            # print(f"brick removed: {min_brick.width} * {min_brick.height}")
            if min_index < len(current_bricks):
                _ = current_bricks.pop(min_index)
                # _ = rotated_bricks_90.pop(min_index)
                _ = rotated_bricks_180.pop(min_index)
               # _ = rotated_bricks_270.pop(min_index)
            elif min_index < 2*len(current_bricks):
                _nb_bricks = len(current_bricks)
                _ = current_bricks.pop(min_index-2*_nb_bricks)
                # _ = rotated_bricks_90.pop(min_index-2*_nb_bricks)
                _ = rotated_bricks_180.pop(min_index-2*_nb_bricks)
                # _ = rotated_bricks_270.pop(min_index-2*_nb_bricks)
            elif min_index < 3*len(current_bricks):
                _nb_bricks = len(current_bricks)
                _ = current_bricks.pop(min_index-3*_nb_bricks)
                _ = rotated_bricks_90.pop(min_index-3*_nb_bricks)
                _ = rotated_bricks_180.pop(min_index-3*_nb_bricks)
            #     _ = rotated_bricks_270.pop(min_index-3*_nb_bricks)
            # else:
            #     _nb_bricks = len(current_bricks)
            #     _ = current_bricks.pop(min_index-4*_nb_bricks)
            #     _ = rotated_bricks_90.pop(min_index-4*_nb_bricks)
            #     _ = rotated_bricks_180.pop(min_index-4*_nb_bricks)
            #     _ = rotated_bricks_270.pop(min_index-4*_nb_bricks)
            # print(len(current_bricks))
        if not current_bricks:
            get_main_logger().warning(
                f"No enough stones to forcast the {i}th step")
            break
        i += 1
        # if k steps reached, break the construction
        if i > get_ksteps():
            break
    return min_distance, base


def _place_once(brick_index, all_stones, all_poses, base, locations):
    brick = all_poses[brick_index]
    # initialization
    min_x = 0
    min_y = 0
    min_distance = evaluate(base, brick)
    # evaluate future distance
    # # build k stones ahead
    # future_base = deepcopy(base)
    # # add current brick to future base
    # future_base.add_stone(brick, [0, 0])
    # # remove current brick from future available stones
    # future_bricks = deepcopy(current_bricks)
    # if not replace:
    #     if brick_index < len(current_bricks):
    #         future_bricks.pop(brick_index)
    #     else:
    #         _nb_bricks = len(current_bricks)
    #         _ = future_bricks.pop(brick_index-_nb_bricks)
    # future_distance = build_kstones(
    #     future_base, future_bricks, i, paras=paras, replace=replace, w_h=weight_height, w_n1=weight_neighbors[0], w_n2=weight_neighbors[1], w_n3=weight_neighbors[2])
    # min_distance += future_distance
    for void_coor in locations:
        try_stone = brick.transformed(
            [void_coor[1], void_coor[0]])
        # if i == 11 and brick_index == 39-len(current_bricks):
        #     print("try_stone:", try_stone)
        if try_stone is None:
            continue
        distance = evaluate(
            base, try_stone)
        # if i == 4 and brick_index == 25 and distance != float("+inf"):
        #     # if void_coor[1] == 0 and void_coor[0] == 26:
        #     print(void_coor[1], void_coor[0], distance)
        # plt.imshow(np.flip(try_stone.matrix, axis=0))
        # plt.show()
        # if i == 14 and brick_index == 10 and distance != float("+inf"):
        #     # if void_coor[1] == 0 and void_coor[0] == 26:
        #     print(void_coor[1], void_coor[0], distance)
        #     plt.imshow(np.flip(try_stone.matrix, axis=0))
        #     plt.show()
        # if void_coor[1] == 56 and void_coor[0] == 16:
        #     print(void_coor[1], void_coor[0], distance)
        #     plt.imshow(np.flip(try_stone.matrix, axis=0))
        #     plt.show()
        # print(distance)
        # if distance != float("+inf"):
        #     print(distance)
        #     print(try_stone.center)
        #     print(void_coor[1], void_coor[0])
        #     try_stone.plot()

        if distance < min_distance:
            min_distance = distance
            min_x = void_coor[1]
            min_y = void_coor[0]
    # reevaluate stone based on height
    if min_distance == float("+inf"):
        get_main_logger().warning(
            "No valid location found for stone {}".format(brick.id))
        return min_x, min_y, min_distance
    try_stone = brick.transformed(
        [min_x, min_y])
    # if i == 1:
    #     #min_distance = 1000/brick.area
    #     local_width = get_local_width()
    #     _base_after = base.matrix+try_stone.matrix
    #     local = _base_after[max(0, int(try_stone.center[1]-try_stone.height/2)-local_width):int(try_stone.center[1] + try_stone.height/2),
    #                         max(0, int(try_stone.center[0]-try_stone.width/2)-local_width):int(try_stone.center[0]+try_stone.width/2)]
    #     nb_void = len(np.argwhere(local == 0))
    #     nb_fill = len(np.argwhere(local != 0))
    #     min_distance = nb_void/(nb_fill)
    # else:

    # try_stone.plot()
    # # ___________________________find the height of left stone
    # placed_stone_centers = np.asarray(base.rock_centers)
    # placed_stone_tops = np.asarray(base.rock_tops)
    # if placed_stone_centers.shape[0] >= 1:
    #     stone_l = np.min(np.nonzero(try_stone.matrix)[1])
    #     stone_b = np.min(np.nonzero(try_stone.matrix)[0])
    #     placed_stone_centers_left = placed_stone_centers[(placed_stone_centers[:, 0]
    #                                                      < stone_l) & (placed_stone_tops > stone_b)]
    #     placed_stone_centers_left_indexs = np.argwhere(
    #         (placed_stone_centers[:, 0] < stone_l) & (placed_stone_tops > stone_b))
    # else:
    #     placed_stone_centers_left = []

    # stone_h = np.max(np.nonzero(try_stone.matrix)[0])
    # if len(placed_stone_centers_left) >= 1:
    #     tree = KDTree(placed_stone_centers_left)
    #     dd, ii = tree.query(try_stone.center, k=1)
    #     #stone_norminal_size = math.sqrt(try_stone.area)
    #     left_stone = base.placed_rocks[placed_stone_centers_left_indexs[ii][0]]
    #     # print(try_stone.center)
    #     # print(placed_stone_centers_left)
    #     # print(dd)
    #     # print(placed_stone_centers)
    #     # print(placed_stone_centers_left_indexs)
    #     # print(placed_stone_centers_left_indexs[ii][0])
    #     left_stone_h = np.max(np.nonzero(left_stone.matrix)[0])

    #     min_distance = abs(stone_h-left_stone_h)/try_stone.height
    # else:
    #     base_b_h = np.max(np.nonzero(
    #         np.where(base.matrix != get_ignore_pixel_value(), base.matrix, 0))[0])
    #     min_distance = 2-abs(stone_h-base_b_h)/try_stone.height
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
        dd, ii = tree.query(query_point, k=1)
        left_stone_h = placed_stone_right_tops_left[ii][1]
        min_distance = abs(stone_h-left_stone_h)
    else:
        heights_stones = np.zeros(len(all_stones))
        for s, stone in enumerate(all_stones):
            heights_stones[s] = stone.height
        base_b_h = np.max(np.nonzero(
            np.where(base.matrix != get_ignore_pixel_value(), base.matrix, 0))[0])
        min_distance = abs(np.median(heights_stones)-try_stone.height)
        # print(np.median(heights_stones))
    rec1 = []
    rec1.append(min_distance)
    # #_test_base_matrix = base.matrix + try_stone.matrix
    # stone_h = np.max(np.nonzero(try_stone.matrix)[0])
    # base_b_h = np.max(np.nonzero(
    #     np.where(base.matrix != get_ignore_pixel_value(), base.matrix, 0))[0])
    # layer_check = (stone_h-base_b_h)/try_stone.height
    # local_width = get_local_width()
    # _base_after = base.matrix+try_stone.matrix
    # local = _base_after[max(0, int(try_stone.center[1]-try_stone.height/2)-local_width):int(try_stone.center[1] + try_stone.height/2),
    #                     max(0, int(try_stone.center[0]-try_stone.width/2)-local_width):int(try_stone.center[0]+try_stone.width/2)]
    # nb_void = len(np.argwhere(local == 0))
    # nb_fill = len(np.argwhere(local != 0))
    # if layer_check > 0.5:  # a new layer
    #     get_main_logger().debug(
    #         "Candidate stone {} is placed on a new course".format(brick.id))
    #     min_distance = 2-(abs(stone_h-base_b_h)/try_stone.height)
    # else:
    #     print(try_stone.id, (abs(stone_h-base_b_h)/try_stone.height))
    #     min_distance = (abs(stone_h-base_b_h)/try_stone.height)
    # print(try_stone.id, 1/try_stone.area)
    min_distance += get_width_weight()*(-np.sqrt(try_stone.area))
    rec1.append(get_width_weight()*(-np.sqrt(try_stone.area)))
    #min_distance += 0.1*abs(1-try_stone.height/try_stone.width)

    # # _______________________________________________________________________interlocking

    # # print(placed_stone_centers.shape)
    # stone_b = np.min(np.nonzero(try_stone.matrix)[0])
    # if placed_stone_centers.shape[0] > 0:
    #     placed_stone_centers_under = placed_stone_centers[placed_stone_centers[:, 1] < stone_b]
    # else:
    #     placed_stone_centers_under = []
    # if len(placed_stone_centers_under) >= 1:
    #     tree = KDTree(placed_stone_centers_under)
    #     dd, ii = tree.query(try_stone.center, k=1)
    #     #stone_norminal_size = math.sqrt(try_stone.area)
    #     horizontal_dist = -abs(
    #         try_stone.center[0]-placed_stone_centers_under[ii][0])
    #     min_distance += get_interlocking_weight()*horizontal_dist/try_stone.width
    #     rec1.append(get_interlocking_weight()*horizontal_dist/try_stone.width)
    # else:
    #     rec1.append(0)
    # _______________________________________________________________________interlocking based on right bottom

    # print(placed_stone_centers.shape)
    placed_stone_left_tops = np.asarray(base.rock_left_tops)
    placed_stone_centers = np.asarray(base.rock_centers)
    stone_b = np.min(np.nonzero(try_stone.matrix)[0])
    if placed_stone_left_tops.shape[0] > 0:
        placed_stone_left_tops_under = placed_stone_left_tops[placed_stone_centers[:, 1] < stone_b]
    else:
        placed_stone_left_tops_under = []
    if len(placed_stone_left_tops_under) >= 1:
        tree = KDTree(placed_stone_left_tops_under)
        query_point = [np.max(np.nonzero(try_stone.matrix)[1]), np.min(
            np.nonzero(try_stone.matrix)[0])]  # max col, min row
        dd, ii = tree.query(query_point, k=1)
        #stone_norminal_size = math.sqrt(try_stone.area)
        horizontal_dist = -abs(
            query_point[0]-placed_stone_left_tops_under[ii][0])
        min_distance += get_interlocking_weight()*(horizontal_dist)
        rec1.append(get_interlocking_weight()*horizontal_dist)
    else:
        rec1.append(0)
    # min_distance = try_stone.center[1]
    # min_distance = 1000/brick.area
    # plt.imshow(np.flip(try_stone.matrix, axis=0))
    # plt.title(f"{min_distance},{stone_h},{base_h}")
    # plt.show()
    # build k stones ahead
    future_base = deepcopy(base)
    # add current brick to future base
    future_base.add_stone(brick, [min_x, min_y])
    # future_base.plot()
    kine = evaluate_kine(future_base)
    if kine > 0:
        min_distance += 1000/kine
    else:
        min_distance = float("+inf")

    return min_x, min_y, min_distance

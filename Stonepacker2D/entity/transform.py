

from skimage.measure import regionprops
import cv2
import matplotlib.pyplot as plt
from Geometry3D import *
import numpy as np
from ..utils.logger import get_main_logger, get_log_level
from ..utils.constant import get_plot_placement, get_plot_stone_id, get_ignore_pixel_value
from skimage.morphology import square,closing,convex_hull_image,opening

def rotate_brick(brick_index, final_base):
    """Rotate the one brick in the base to make it stable

    :param brick_index: Brick index in the order of placement
    :type brick_index: int
    :param final_base: A base containing placed stones
    :type final_base: Base object
    :return: Whether the stone is stable or not. A stone is stable if its center of mass is envelopped by contact with under stones with or without rotation
    :rtype: bool
    """
    other_stones = np.zeros(final_base.matrix.shape)
    stone_index = brick_index
    other_stones = final_base.matrix - \
        final_base.placed_bricks[stone_index].matrix*(stone_index+1)
    #other_stones = base.copy()
    # find stones below
    indices = []
    stone_prop = regionprops(
        final_base.placed_bricks[stone_index].matrix.astype(np.uint8))
    low_bound = stone_prop[0].bbox[0]
    for index, brick in enumerate(final_base.placed_bricks):
        other_stone_prop = regionprops(brick.matrix.astype(np.uint8))
        if other_stone_prop[0].centroid[0] < low_bound:
            indices.append(index)
    base = np.zeros(final_base.matrix.shape)

    for index in indices:
        base += final_base.placed_bricks[index].matrix*(index+1)

    # find the contact by dilating 1 pixel
    img = final_base.placed_bricks[stone_index].matrix.astype(np.uint8)
    if get_log_level() == 'DEBUG':
        plt.imshow(img)
        plt.show()
        plt.imshow(base)
        plt.show()
    kernel_dilation_y = np.ones((3, 1), np.uint8)

    img = cv2.dilate(img, kernel_dilation_y, iterations=1)
    # if there's no stone below=> the first layer
    if len(indices) == 0:
        get_main_logger().debug(
            "No stone is found beneath the stone when the stone is dilated one pixel")
        intersection = img[0, :].reshape((1, -1))  # take the floor pixels
    else:
        intersection = np.multiply(img, base)
        # if no intersection is found
        if np.argwhere(intersection).size == 0:
            get_main_logger().debug(
                "No intersection is found of the stone and the stones underneath, the stone is not stable")
            return False
    if np.argwhere(intersection).size == 0:
        return False
    intersection_prop = regionprops(intersection.astype(np.uint8))
    if not intersection_prop:  # region prop fails
        if np.argwhere(intersection).shape[0]>1:
            min_col = np.min(np.argwhere(intersection), axis=0)[1]
            max_col = np.max(np.argwhere(intersection), axis=0)[1]
            if max_col == min_col:
                max_col += 1
            row = np.argwhere(intersection)[0][0]
        else:
            min_col = np.argwhere(intersection)[1]
            max_col = min_col+1
            row = np.argwhere(intersection)[0]

    else:
        min_col = intersection_prop[0].bbox[1]
        max_col = intersection_prop[0].bbox[3]
        row = intersection_prop[0].bbox[2]
    # find center of rotation according to relative position of center
    if stone_prop[0].centroid[1] > min_col and stone_prop[0].centroid[1] < max_col-1:
        # print("stable")
        rot_center = None
    elif stone_prop[0].centroid[1] <= min_col:  # rotate around row, min_col
        # rot_center = (row, min_col)
        rot_center = (min_col, row)
        mask = np.zeros(final_base.matrix.shape)
        mask[:, 0:int(stone_prop[0].centroid[1])] = 1
        rotate_clockwise = False
    elif stone_prop[0].centroid[1] >= max_col-1:
        # rot_center = (row, max_col)
        rot_center = (max_col-1, row)
        mask = np.zeros(final_base.matrix.shape)
        mask[:, int(stone_prop[0].centroid[1]):] = 1
        rotate_clockwise = True
    if rot_center is not None:
        if not rotate_clockwise:
            _min_angle = 0
            _max_angle = -5
            _delta_angle = -1
        else:
            _max_angle = 5
            _min_angle = 0
            _delta_angle = 1
        img = final_base.placed_bricks[stone_index].matrix.astype(np.uint8)
        for angle in range(_min_angle+_delta_angle, _max_angle, _delta_angle):
            # angle is positive for anti-clockwise, but image is flipped in terms of coordinates
            # print("Rotation center is",rot_center)
            # print("Angle is",angle)
            rot_mat = cv2.getRotationMatrix2D(rot_center, angle, 1.0)
            img_rotated = cv2.warpAffine(
                img, rot_mat, img.shape[1::-1], flags=cv2.WARP_FILL_OUTLIERS)
            if len(np.argwhere(img_rotated)) == 0:
                return False
            # check penetration
            product = np.multiply(img_rotated, other_stones)
            # plt.imshow(product)
            # plt.title("not rotate because of penetration")
            # plt.show()
            if np.any(product):
                rot_mat = cv2.getRotationMatrix2D(
                    rot_center, angle-_delta_angle, 1.0)
                img_rotated = cv2.warpAffine(
                    img, rot_mat, img.shape[1::-1], flags=cv2.WARP_FILL_OUTLIERS)
                break

            # check contact with the other side
            # apply mask to base
            if mask is not None:
                base = np.multiply(base, mask)
            img_rotated_dilated = cv2.dilate(
                img_rotated, kernel_dilation_y, iterations=1)
            # if there's no stone below=> the first layer
            if len(indices) == 0:
                get_main_logger().warning(
                    "No stone is found beneath the stone when trying to rotate the stone")
                if mask is not None:
                    intersection = np.multiply(img_rotated_dilated, mask)[
                        0, :].reshape((1, -1))
                else:
                    intersection = img_rotated_dilated[0, :].reshape((1, -1))
                if np.sum(intersection, axis=1)[0] != 0:
                    # plt.imshow(intersection)
                    # plt.title("not rotate because of intersection formed")
                    # plt.show()
                    # plt.imshow(intersection)
                    # plt.show()
                    break
            else:
                intersection = np.multiply(img_rotated_dilated, base)
                if np.argwhere(intersection).size > 0:
                    # plt.imshow(intersection)
                    # plt.show()
                    break
        if angle == -(_max_angle-_delta_angle):
            get_main_logger().warning(
                "The stone cannot be stabalized by rotating")
            return False
        else:
            # final_base.plot()
            # plt.imshow(np.flip(img, axis=0))
            # plt.show()
            # plt.imshow(np.flip(img_rotated, axis=0))
            # plt.title(f"rotated ,angle{angle}")
            # plt.show()

            #img_rotated = cv2.medianBlur(img_rotated.astype(np.uint8), 3)
            img_rotated = opening(img_rotated, square(3))
            #img_rotated = convex_hull_image(img_rotated,offset_coordinates=False)
            img_rotated = np.where(np.multiply(
                img_rotated, other_stones) == 0, img_rotated, 0)
            if len(np.argwhere(img_rotated)) == 0:
                return False

            final_base.matrix = final_base.matrix - \
                final_base.placed_bricks[stone_index].matrix * \
                (stone_index+1)+img_rotated*(stone_index+1)
            final_base.id_matrix = final_base.id_matrix - \
                final_base.placed_bricks[stone_index].matrix * \
                (final_base.placed_bricks[stone_index].id) + \
                img_rotated*(final_base.placed_bricks[stone_index].id)
            final_base.cluster_matrix = final_base.cluster_matrix - \
                final_base.placed_bricks[stone_index].matrix * \
                (final_base.placed_bricks[stone_index].cluster) + \
                img_rotated*(final_base.placed_bricks[stone_index].cluster)
            final_base.centers[stone_index] = [
                stone_prop[0].centroid[1], stone_prop[0].centroid[0]]
            final_base.placed_bricks[stone_index].matrix = img_rotated
            final_base.placed_bricks[stone_index].rot_center = rot_center
            final_base.placed_bricks[stone_index].rot_angle += angle

            #!dangerous
            if final_base.placed_bricks[stone_index].id !=final_base.placed_rocks[-1].id:
                #raise errors
                get_main_logger().error("Rotation stabalization: The id of the stone is not the same as the id of the last placed rock")
            final_base.placed_rocks[-1].matrix = img_rotated
            final_base.placed_rocks[-1].rot_center = rot_center
            final_base.placed_rocks[-1].rot_angle += angle

            # final_base.plot()
            get_main_logger().debug(
                f"Stone {final_base.placed_bricks[stone_index].id}")
            return True
    else:
        return True


def check_stable(stone_index, final_base):
    # find stones below
    indices = []
    stone_prop = regionprops(
        final_base.placed_bricks[stone_index].matrix.astype(np.uint8))
    if not stone_prop:
        return True
    low_bound = stone_prop[0].bbox[0]
    for index, brick in enumerate(final_base.placed_bricks):
        other_stone_prop = regionprops(brick.matrix.astype(np.uint8))
        if other_stone_prop[0].centroid[0] < low_bound:
            indices.append(index)
    base = np.zeros(final_base.matrix.shape)

    for index in indices:
        base += final_base.placed_bricks[index].matrix*(index+1)

    # find the contact by dilating 1 pixel
    img = final_base.placed_bricks[stone_index].matrix.astype(np.uint8)
    if get_log_level() == 'DEBUG':
        plt.imshow(img)
        plt.show()
        plt.imshow(base)
        plt.show()
    kernel_dilation_y = np.ones((3, 1), np.uint8)

    img = cv2.dilate(img, kernel_dilation_y, iterations=1)
    # if there's no stone below=> the first layer
    if len(indices) == 0:
        get_main_logger().debug(
            "No stone is found beneath the stone when the stone is dilated one pixel")
        intersection = img[0, :].reshape((1, -1))  # take the floor pixels
    else:
        intersection = np.multiply(img, base)
        # if no intersection is found
        if np.argwhere(intersection).size == 0:
            get_main_logger().debug(
                "No intersection is found of the stone and the stones underneath, the stone is not stable")
            return False

    intersection_prop = regionprops(intersection.astype(np.uint8))
    if not intersection_prop:  # region prop fails
        min_col = np.min(np.argwhere(intersection), axis=0)[1]
        max_col = np.max(np.argwhere(intersection), axis=0)[1]
        if max_col == min_col:
            max_col += 1
        row = np.argwhere(intersection)[0][0]
    else:
        min_col = intersection_prop[0].bbox[1]
        max_col = intersection_prop[0].bbox[3]
        row = intersection_prop[0].bbox[2]
    # find center of rotation according to relative position of center
    if stone_prop[0].centroid[1] >= min_col and stone_prop[0].centroid[1] <= max_col:
        # print("stable")
        return True
    return False

def move_down(stone, final_base,optimization_y_scale = [0,0]):
    # check overlap
    stone_matrix = stone.matrix
    base_matrix = np.where(final_base.matrix!=get_ignore_pixel_value(), final_base.matrix, 0)
    step=0
    while(np.argwhere(np.multiply(stone_matrix, base_matrix)).size!=0):
        # # if overlap, move to right
        # stone = stone.transformed([1,0])
        # stone_matrix = stone.matrix
        # step+=1
        # if step%1==0:
        #     stone = stone.transformed([0,1])
        #     stone_matrix = stone.matrix
        # if overlap, remove overlapped pixels
        stone.matrix = np.where(np.multiply(stone_matrix, base_matrix)==0, stone_matrix, 0)
        stone_matrix = stone.matrix
   
    # move down until overlap
    #check if matrix is all zero
    if len(np.argwhere(stone_matrix)) == 0:
        return stone
    stone_max_y = np.argwhere(stone_matrix!=0)[:,0].max()
    translation_y = 0
    while((np.argwhere(np.multiply(stone_matrix, base_matrix)).size==0)&(translation_y<(stone_matrix.shape[0]-stone_max_y)))and translation_y<-optimization_y_scale[0]:
        stone = stone.transformed([0,-1])
        stone_matrix = stone.matrix
        stone_max_y = np.argwhere(stone_matrix!=0)[:,0].max()
        translation_y+=1
        # print("Translation y: ", translation_y)
    
    stone = stone.transformed([0,1])
    stone.displacement_for_ka[1] = -translation_y
    return stone

def move_left(stone, final_base,optimization_x_scale = [0,0]):
    # check overlap
    stone_matrix = stone.matrix
    base_matrix = np.where(final_base.matrix!=get_ignore_pixel_value(), final_base.matrix, 0)
    step=0
    while(np.argwhere(np.multiply(stone_matrix, base_matrix)).size!=0):
        # # if overlap, move to right
        # stone = stone.transformed([1,0])
        # stone_matrix = stone.matrix
        # step+=1
        # if step%1==0:
        #     stone = stone.transformed([0,1])
        #     stone_matrix = stone.matrix
        # if overlap, remove overlapped pixels
        stone.matrix = np.where(np.multiply(stone_matrix, base_matrix)==0, stone_matrix, 0)
        stone_matrix = stone.matrix
   
    # move down until overlap
    #check if matrix is all zero
    if np.argwhere(stone_matrix).size == 0:
        return stone
    stone_min_x = np.argwhere(stone_matrix!=0)[:,1].min()
    translation_x = 0
    while((np.argwhere(np.multiply(stone_matrix, base_matrix)).size==0)&(stone_min_x>0)) and translation_x<-optimization_x_scale[0]:
        stone = stone.transformed([-1,0])
        stone_matrix = stone.matrix
        stone_min_x = np.argwhere(stone_matrix!=0)[:,1].min()
        translation_x+=1
        #print("Translation x: ", translation_x)
    
    stone = stone.transformed([1,0])
    stone.displacement_for_ka[0] = -translation_x
    # plt.imshow(stone.matrix)
    # plt.show()
    return stone
from ..evaluate.local_void import evaluate
from pyMetaheuristic.algorithm import flow_direction_algorithm
def isNaN(num):
    return num != num
def optimize_move(stone, final_base,optimization_x_scale = [0,0],optimization_y_scale = [0,0]):
    # check overlap
    stone_matrix = stone.matrix
    base_matrix = np.where(final_base.matrix!=get_ignore_pixel_value(), final_base.matrix, 0)
    stone.matrix = np.where(np.multiply(stone_matrix, base_matrix)==0, stone_matrix, 0)
    stone_matrix = stone.matrix

    #optimize such that local void is minimized
    def local_void(variables_values = [0, 0]):
        if isNaN(variables_values[0]) or isNaN(variables_values[1]):
            return float("+inf")
        try_stone = stone.transformed(
            [int(variables_values[0]), int(variables_values[1])])
        if try_stone is None:
            return float("+inf")
        if len(np.argwhere(try_stone.matrix!=0)) == 0:
            return float("+inf")
        stone_min_x = np.argwhere(try_stone.matrix!=0)[:,1].min()
        if stone_min_x<=0:
            return float("+inf")
        
        else:
            distance,_,_ = evaluate(
                final_base, try_stone)
            return distance
    
    parameters = {
            'size': 5,
            'min_values': (optimization_x_scale[0], optimization_y_scale[0]),
            'max_values': (optimization_x_scale[1], optimization_y_scale[1]),
            'iterations': 3,
            'beta': 8,#number of neighbors
            'verbose': False
        }
    fda = flow_direction_algorithm(target_function = local_void, **parameters)
    if fda[-1]==float("+inf"):
        return stone
    min_x = int(fda[0])
    min_y = int(fda[1])

    stone = stone.transformed([min_x,min_y])
    stone.displacement_for_ka[0]+=min_x
    stone.displacement_for_ka[1]+=min_y
    # plt.imshow(stone.matrix)
    # plt.show()
    return stone
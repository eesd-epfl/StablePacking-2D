
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import cv2
import numpy as np
from .brick import Wedge
from ..utils.logger import get_main_logger
from ..utils.constant import get_plot_placement, get_plot_stone_id


def find_wedge(stone_index, final_base, mask=None,max_wedge_width = 20,max_wedge_height = 10):
    stone_index = stone_index
    base = final_base.matrix - \
        final_base.placed_bricks[stone_index].matrix*(stone_index+1)
    # find stones below
    indices = []
    stone_prop = regionprops(
        final_base.placed_bricks[stone_index].matrix.astype(np.uint8))
    low_bound = stone_prop[0].bbox[0]
    for index, brick in enumerate(final_base.placed_bricks):
        other_stone_prop = regionprops(brick.matrix.astype(np.uint8))
        if not other_stone_prop:
            if len(np.argwhere(brick.matrix)) == 0:
                # plt.imshow(brick.matrix)
                print(brick.id)
                # plt.show()
            min_row = np.min(np.argwhere(brick.matrix), axis=0)[0]
            max_row = np.max(np.argwhere(brick.matrix), axis=0)[0]
            other_center_y = 0.5*(min_row+max_row)
        else:
            other_center_y = other_stone_prop[0].centroid[0]
        if other_center_y < low_bound:
            indices.append(index)
    base = np.zeros(final_base.matrix.shape)

    for index in indices:
        base += final_base.placed_bricks[index].matrix*(index+1)
    # apply mask
    if mask is not None:
        base = np.multiply(base, mask)

    # dilate stone and find first intersection
    _max_dilation = 200
    img = final_base.placed_bricks[stone_index].matrix.astype(np.uint8)
    kernel_dilation_y = np.ones((3, 1), np.uint8)
    # plt.imshow(np.flip(img, axis=0))
    # plt.title(f'original stone N {stone_index}')
    # plt.xticks(np.arange(0, 10, step=1))
    # plt.yticks(np.arange(0, 10, step=1))
    # plt.show()
    for dilate in range(1, _max_dilation, 1):
        img = cv2.dilate(img, kernel_dilation_y, iterations=1)
        # plt.imshow(np.flip(img, axis=0))
        # plt.title(f'dilate {dilate}')
        # plt.xticks(np.arange(0, 10, step=1))
        # plt.yticks(np.arange(0, 10, step=1))
        # plt.show()
        # if there's no stone below=> the first layer
        if len(indices) == 0:
            if mask is not None:
                intersection = np.multiply(img, mask)[0, :].reshape((1, -1))
            else:
                intersection = img[0, :].reshape((1, -1))
            if np.sum(intersection, axis=1)[0] != 0:
                # plt.imshow(intersection)
                # plt.show()
                break
        else:
            intersection = np.multiply(img, base)
            if np.argwhere(intersection).size > 0:
                # plt.imshow(intersection)
                # plt.show()
                break
    # if no intersection is found
    if np.argwhere(intersection).size == 0:
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(base)
        # plt.show()
        # if mask is not None:
        #     plt.imshow(mask)
        #     plt.show()
        return np.zeros(base.shape), None
    intersection_prop = regionprops(intersection.astype(np.uint8))
    # print(np.argwhere(intersection))
    # print(intersection_prop)
    # plt.imshow(img)
    # plt.show()
    # plt.imshow(base)
    # plt.show()
    if not intersection_prop:  # region prop fails
        min_col = np.min(np.argwhere(intersection), axis=0)[1]
        max_col = np.max(np.argwhere(intersection), axis=0)[1]+1
        if max_col == min_col:
            max_col += 1
        transformation = [
            np.min(np.argwhere(intersection), axis=0)[0], min_col]
    else:
        min_col = intersection_prop[0].bbox[1]
        max_col = intersection_prop[0].bbox[3]
        transformation = [intersection_prop[0].bbox[2], min_col]
    wedge_height = dilate-1
    wedge_width = max_col-min_col
    if wedge_height>max_wedge_height or wedge_width>max_wedge_width or wedge_height/wedge_width>max_wedge_height/max_wedge_width:
        return np.zeros(base.shape), None
    # if wedge_height > 5:
    #     return np.zeros(base.shape), None
    
    # if np.any(np.argwhere(np.multiply(wedge, img))):
    #     return np.zeros(base.shape), None
    # #cheating
    # wedge_height+=2
    # wedge_width+=2
    # transformation[0]-=1
    # transformation[1]-=1

    wedge = np.zeros(base.shape)
    wedge[transformation[0]:transformation[0]+wedge_height,
          transformation[1]:transformation[1]+wedge_width] = len(final_base.centers)+1
          
    if stone_prop[0].centroid[1] > min_col and stone_prop[0].centroid[1] < max_col:
        if len(np.argwhere(wedge != 0)) != 0:
            get_main_logger().critical(
                f'Add wedge (width {wedge_width}, height {wedge_height}) to stone {final_base.placed_bricks[stone_index].id}')
            final_base.add_stone(Wedge(wedge_width, wedge_height), [
                                 transformation[1], transformation[0]])
        return wedge, None
    # if the stone center is on the left of the bbox
    elif stone_prop[0].centroid[1] <= min_col:
        if len(np.argwhere(wedge != 0)) != 0:
            get_main_logger().critical(
                f'Add wedge (width {wedge_width}, height {wedge_height}) to stone {final_base.placed_bricks[stone_index].id}')
            final_base.add_stone(Wedge(wedge_width, wedge_height), [
                                 transformation[1], transformation[0]])

        mask = np.zeros(final_base.matrix.shape)
        mask[:, 0:int(stone_prop[0].centroid[1])+1] = 1
        return wedge, mask
    # if the stone center is on the right of the bbox
    elif stone_prop[0].centroid[1] >= max_col:
        if len(np.argwhere(wedge != 0)) != 0:
            get_main_logger().critical(
                f'Add wedge (width {wedge_width}, height {wedge_height}) to stone {final_base.placed_bricks[stone_index].id}')
            final_base.add_stone(Wedge(wedge_width, wedge_height), [
                                 transformation[1], transformation[0]])

        mask = np.zeros(final_base.matrix.shape)
        mask[:, int(stone_prop[0].centroid[1]):] = 1
        return wedge, mask


def add_wedges(stone_indexs, final_base):
    wedges = np.zeros((final_base.matrix.shape))
    for stone_index in stone_indexs:
        #stone_index = 13
        #print(f"stone index {stone_index}")
        mask = None
        nb_wedge = 0
        counter = 0
        while True:
            wedge, mask = find_wedge(
                stone_index, final_base, mask=mask)
            wedges += wedge
            counter += 1
            if len(np.argwhere(wedge != 0)) != 0:
                nb_wedge += 1
            if mask is None or counter == 2:
                # if len(final_base.placed_bricks) == int(get_plot_placement()+nb_wedge+1) and final_base.placed_bricks[stone_index].id == get_plot_stone_id():
                #     plt.clf()
                #     plt.imshow(
                #         np.flip(final_base.placed_bricks[stone_index].matrix+wedges, axis=0), cmap='gray')
                #     plt.savefig(
                #         f"/home/qiwang/Projects/15_Stonepacker2D/img/stone_wedge{final_base.placed_bricks[stone_index].id}.png")

                # plt.clf()
                # plt.imshow(
                #     np.flip(final_base.placed_bricks[stone_index].matrix, axis=0), cmap='gray')
                # plt.savefig(
                #     "/home/qiwang/Projects/15_Stonepacker2D/img/stone_wedge_0.png")

                # plt.clf()
                # plt.imshow(
                #     np.flip(wedges, axis=0), cmap='gray')
                # plt.savefig(
                #     f"/home/qiwang/Projects/15_Stonepacker2D/img/stone_wedge_1{final_base.placed_bricks[stone_index].id}.png")
                break

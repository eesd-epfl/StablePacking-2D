import numpy as np
import math


def evaluate_first(base, stone, weight_height=1, weight_first_to_origin_x=0):
    # check penetration
    product = np.multiply(base.matrix, stone.matrix)

    if np.any(product):
        return float("+inf")
    # calculate distance to the bottom of the base
    ditance = abs(stone.center[1])

    # calculate distance to the original
    ditance += weight_first_to_origin_x*abs(stone.center[0])

    # normalize distance
    stone_norminal_size = math.sqrt(stone.area)
    return ditance/stone_norminal_size

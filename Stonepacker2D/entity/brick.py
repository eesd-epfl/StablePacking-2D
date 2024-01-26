from skimage.measure import regionprops
import matplotlib.pyplot as plt
from Geometry3D import *
import numpy as np
from ..utils.constant import get_world_size, get_wedge_id
from .stone import Stone


class Brick(Stone):
    """A base class for stones, based on rectangular shape
    """

    def __init__(self, width, height):
        """Constructor of a rectangular shaped stone

        :param width: Width of the brick along the x axis
        :type width: float
        :param height: Height of the brick along the y axis
        :type height: float
        """
        self.center = [width/2, height/2]
        self.width = width
        self.height = height
        _w, _h = get_world_size()
        self.matrix = np.zeros((_h, _w))
        # inside the brick, pixels' value is -1
        self.matrix[0:height,
                    0:width] = 1
        self.area = self.width*self.height
        self._w = _w
        self._h = _h
        self.rot_center = (0, 0)  # x,y
        self.rot_angle = 0  # in degree
        self.displacement_for_ka = [0,0]
        self.vertices = None
        self.id = 1
        self.cluster = -1
        self.shape_factor = None
        self.roundness = None

        self.rotate_from_ori = 0

    # def transformed(self, transformation):
    #     """Translate the brick to a new position

    #     :param transformation: Transformation in x and y axis
    #     :type transformation: list
    #     :return: Translated brick
    #     :rtype: Brick object
    #     """
    #     # check bounding box
    #     if self.center[0]-self.width/2+transformation[0] < 0 or round(self.center[0]+self.width/2+transformation[0]) > self._w or \
    #             self.center[1]-self.height/2+transformation[1] < 0 or round(self.center[1]+self.height/2+transformation[1]) > self._h:
    #         return None
    #     # if round(self.center[1]+transformation[1]-self.height/2) >= self._w or round(self.center[0]+transformation[0]-self.width/2) >= self._h:
    #     #     return None
    #     # transform the brick
    #     transformed_brick = Brick(self.width, self.height)
    #     transformed_brick.center[0] = self.center[0]+transformation[0]
    #     transformed_brick.center[1] = self.center[1]+transformation[1]
    #     transformed_brick.matrix = np.zeros(
    #         (self._h, self._w))
    #     transformed_brick.matrix[round(transformed_brick.center[1]-transformed_brick.height/2):round(transformed_brick.center[1]+transformed_brick.height/2),
    #                              round(transformed_brick.center[0]-transformed_brick.width/2):round(transformed_brick.center[0]+transformed_brick.width/2)] = 1
    #     transformed_brick.id = self.id
    #     return transformed_brick

    # def rotated_by_90(self):
    #     """Rotate the brick by 90 degree

    #     :return: Rotated brick
    #     :rtype: Brick object
    #     """
    #     # rotate the brick
    #     # initialize around the origin point
    #     brick = Brick(self.height, self.width)
    #     brick.id = self.id
    #     return brick

    def plot(self):
        """Plot the brick
        """
        # plot Matrix
        plt.imshow(np.flip(self.matrix, axis=0))
        plt.show()

    def __add__(self,other):
        self.matrix+=other.matrix
        region = regionprops(self.matrix.astype(np.uint8))
        self.center = [region[0].centroid[1], region[0].centroid[0]]
        self.width = region[0].bbox[3]-region[0].bbox[1]
        self.height = region[0].bbox[2]-region[0].bbox[0]
        return self

class Wedge(Stone):
    def __init__(self, width, height):
        """Constructor of a rectangular shaped stone

        :param width: Width of the brick along the x axis
        :type width: float
        :param height: Height of the brick along the y axis
        :type height: float
        """
        self.center = [width/2, height/2]
        self.width = width
        self.height = height
        _w, _h = get_world_size()
        self.matrix = np.zeros((_h, _w))
        # inside the brick, pixels' value is -1
        self.matrix[0:height,
                    0:width] = 1
        self.area = self.width*self.height
        self._w = _w
        self._h = _h
        self.rot_center = (0, 0)  # x,y
        self.rot_angle = 0  # in degree
        self.displacement_for_ka = [0,0]
        self.vertices = None
        self.id = get_wedge_id()
        self.roundness = None
        self.cluster = -1
        self.rotate_from_ori = 0

from skimage.measure import regionprops
import cv2
import matplotlib.pyplot as plt
from Geometry3D import *
import numpy as np
import math
from ..utils.constant import get_world_size, get_plot_stone_id
from ..utils.logger import get_main_logger
import skimage.transform as transform
from skimage.morphology import square,closing,convex_hull_image,opening
def _cal_center(points):
    """
    Calculate the averaged center of a list of points
    :param points: list of points
    :type points: list
    :return: center of the polygon
    :rtype: list
    """
    x = 0
    y = 0
    for p in points:
        x += p[0]
        y += p[1]
    return [x/len(points), y/len(points)]


class Stone():
    """Class for irregular stones
    """

    def __init__(self):
        """Constructor
        """
        self.center = None
        self.matrix = None
        self.area = None

        self.width = None  # bounding box width
        self.height = None  # bounding box height

        self.rot_center = (0, 0)  # x,y
        self.rot_angle = 0  # in degree
        self.displacement_for_ka = [0,0]
        self.vertices = None
        self._w, self._h = get_world_size()
        self.id = None

        self.shape_factor = None
        self.roundness = None
        self.cluster = -1

        self.rotate_from_ori = 0
    
    def create_mirrored_stone(self, align_to_right = False):
        """Create a mirrored stone

        :return: Mirrored stone
        :rtype: Stone
        """
        mirrored_stone = Stone()
        mirrored_stone.matrix = np.flip(self.matrix, axis=1)
        mirrored_stone.center = [self.matrix.shape[1]-self.center[0], self.center[1]]
        mirrored_stone.area = self.area
        mirrored_stone.width = self.width
        mirrored_stone.height = self.height
        mirrored_stone.rot_center = [self.matrix.shape[1]-self.rot_center[0], self.rot_center[1]]
        mirrored_stone.rot_angle = -self.rot_angle
        mirrored_stone.vertices = self.vertices
        mirrored_stone._w, mirrored_stone._h = self._w, self._h
        mirrored_stone.id = self.id
        mirrored_stone.shape_factor = self.shape_factor
        mirrored_stone.roundness = self.roundness
        mirrored_stone.cluster = self.cluster
        mirrored_stone.rotate_from_ori = self.rotate_from_ori

        if align_to_right:
            mirrored_stone.align_to_right()
        return mirrored_stone

    def align_to_right(self):
        # crop the flipped matrix
        stone_pixels = np.argwhere(self.matrix)
        top_left = stone_pixels.min(axis=0)
        bottom_right = stone_pixels.max(axis=0)
        cropped_matrix = self.matrix[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
        # plt.imshow(ori_tranformed_matrix)
        # plt.show()
        aligned_matrix = np.zeros((self._h, self._w))
        aligned_matrix[:cropped_matrix.shape[0],
                          :cropped_matrix.shape[1]] = cropped_matrix
        self.matrix = aligned_matrix
        self.center = [self.matrix.shape[1]-self.center[0], self.center[1]]
        self.rot_center = [self.matrix.shape[1]-self.rot_center[0], self.rot_center[1]]
        self.rot_angle = -self.rot_angle
    
    
        

    def from_labeled_img(self, label, tif, scale=1,new_label = None):
        """Read a labeled stone from image.

        :param label: Label of the stone in the image
        :type label: int, from 1 to 255 (0 is always the background in the image)
        :param tif: Path to the image
        :type tif: str
        :return: Whether reading is successful
        :rtype: bool
        """
        
        # crop the matrix
        #matrix_from_img = np.flip(cv2.imread(tif,cv2.IMREAD_GRAYSCALE), axis=0)
        matrix_from_img = cv2.imread(tif,cv2.IMREAD_GRAYSCALE)
        # plt.imshow(matrix_from_img)
        # plt.show()        
        # rescale image
        dim_0 = int(matrix_from_img.shape[1]*scale)
        dim_1 = int(matrix_from_img.shape[0]*scale)
        dim = (dim_0, dim_1)
        matrix_from_img = cv2.resize(
            matrix_from_img, dim, interpolation=cv2.INTER_NEAREST)

        true_stone = np.argwhere(matrix_from_img == label)
        # print(true_stone)
        if true_stone.any():
            stone_matrix_ori_size = np.zeros(matrix_from_img.shape)
            #print("stone_matrix_ori_size, matrix_from_img shape: ", matrix_from_img.shape)
            stone_matrix_ori_size[matrix_from_img == label] = 1
            top_left = true_stone.min(axis=0)
            bottom_right = true_stone.max(axis=0)
            #print("min bound",top_left)
            #print("max bound",bottom_right)
            cropped_matrix = stone_matrix_ori_size[top_left[0]
                : bottom_right[0]+1, top_left[1]: bottom_right[1]+1]
            #print("cropped matrix shape: ", cropped_matrix.shape)
            
            

            self.matrix = np.zeros((self._h, self._w))
            # check if stone is too large
            if cropped_matrix.shape[0]>self._h or cropped_matrix.shape[1]>self._w:
                print(f"stone {new_label} is too large, skip")
                return False
            #print("self matrix shape: ", self.matrix.shape)
            self.matrix[: cropped_matrix.shape[0],
                        : cropped_matrix.shape[1]] = cropped_matrix

            center = regionprops(self.matrix.astype(np.uint8))[0].centroid
            self.center = [center[1], center[0]]
            # plt.imshow(self.matrix)
            # plt.show()
            self.area = np.count_nonzero(self.matrix)
            # check crop
            if (self.area) != np.count_nonzero(stone_matrix_ori_size):
                get_main_logger().error("Stone being cropped")
                raise TypeError("Bug detected! please contact the author")
            region_prop = regionprops(self.matrix.astype(np.uint8))
            self.width = region_prop[0].bbox[3]-region_prop[0].bbox[1]
            self.height = region_prop[0].bbox[2]-region_prop[0].bbox[0]
            if new_label is not None:
                self.id = new_label
            else:
                self.id = label
            self.roundness = region_prop[0].eccentricity
            return True
        else:
            return False

    def from_matrix(self, matrix,scale=1):
        # rescale image
        dim_0 = int(matrix.shape[1]*scale)
        dim_1 = int(matrix.shape[0]*scale)
        dim = (dim_0, dim_1)
        matrix = cv2.resize(
                matrix, dim, interpolation=cv2.INTER_NEAREST)
        true_stone = np.argwhere(matrix !=0)
        if true_stone.any():
            stone_matrix_ori_size = np.zeros(matrix.shape)
            stone_matrix_ori_size[matrix !=0] = 1
            top_left = true_stone.min(axis=0)
            bottom_right = true_stone.max(axis=0)
            cropped_matrix = stone_matrix_ori_size[top_left[0]
                : bottom_right[0]+1, top_left[1]: bottom_right[1]+1]
            # cropped_matrix = np.flip(cropped_matrix, axis=0)
            self.matrix = np.zeros((self._h, self._w))
            self.matrix[: cropped_matrix.shape[0],
                        : cropped_matrix.shape[1]] = cropped_matrix
            center = regionprops(self.matrix.astype(np.uint8))[0].centroid
            self.center = [center[1], center[0]]
            # plt.imshow(self.matrix)
            # plt.show()
            self.area = np.count_nonzero(self.matrix)
            # check crop
            if (self.area) != np.count_nonzero(stone_matrix_ori_size):
                get_main_logger().error("Stone being cropped")
                raise TypeError("Bug detected! please contact the author")
            region_prop = regionprops(self.matrix.astype(np.uint8))
            self.width = region_prop[0].bbox[3]-region_prop[0].bbox[1]
            self.height = region_prop[0].bbox[2]-region_prop[0].bbox[0]
            self.id = matrix[true_stone[0][0], true_stone[0][1]].astype(int)
            self.roundness = region_prop[0].eccentricity
            return True,(top_left[1],top_left[0])
        else:
            return False,None

    # def convert_poly_to_stone(self, points):
    #     self.center = _cal_center(points)
    #     _w, _h = get_world_size()
    #     self._w = _w
    #     self._h = _h
    #     img = np.zeros((_h, _w, 3))
    #     img_with_stone = cv2.fillPoly(
    #         img, pts=[points], color=(1, 1, 1))
    #     # for p in points:
    #     #     plt.scatter(p[0], p[1])
    #     # plt.imshow(img_with_stone)
    #     # plt.show()
    #     # self.matrix = img_with_stone[:, :, 0]*(1)
    #     matrix_from_img = img_with_stone[:, :, 0]*(1)
    #     true_stone = np.argwhere(matrix_from_img)
    #     top_left = true_stone.min(axis=0)
    #     cropped_matrix = matrix_from_img[top_left[0]:, top_left[1]:]
    #     self.matrix = np.zeros((_h, _w))
    #     self.matrix[:cropped_matrix.shape[0],
    #                 :cropped_matrix.shape[1]] = cropped_matrix
    #     self.center = [self.center[0]-top_left[1], self.center[1]-top_left[0]]
    #     # plt.imshow(self.matrix)
    #     # plt.show()
    #     self.area = np.count_nonzero(self.matrix)
    #     # check crop
    #     if (self.area) != np.count_nonzero(matrix_from_img):
    #         raise Exception("invalid cropped stone")
    #     region_prop = regionprops(self.matrix.astype(np.uint8))
    #     bbx_width = region_prop[0].bbox[3]-region_prop[0].bbox[1]
    #     bbx_hight = region_prop[0].bbox[2]-region_prop[0].bbox[0]

    #     self.height = int(math.sqrt(self.area/(bbx_width/bbx_hight)))
    #     self.width = int((bbx_width/bbx_hight)*self.height)
    #     print(self.width)
    #     # matrix_from_img = img_with_stone[:, :, 0]*(1)
    #     # true_stone = np.argwhere(matrix_from_img)
    #     # top_left = true_stone.min(axis=0)
    #     # cropped_matrix = matrix_from_img[top_left[0]:, top_left[1]:]
    #     # self.matrix_shape = np.zeros((_h, _w))
    #     # self.matrix_shape[:cropped_matrix.shape[0],
    #     #                   :cropped_matrix.shape[1]] = cropped_matrix
    #     # self.center = [self.center[0]-top_left[1], self.center[1]-top_left[0]]
    #     # # plt.imshow(self.matrix)
    #     # # plt.show()
    #     # self.area = np.count_nonzero(self.matrix_shape)
    #     # # check crop
    #     # if (self.area) != np.count_nonzero(matrix_from_img):
    #     #     raise Exception("invalid cropped stone")

    #     # region_prop = regionprops(self.matrix_shape.astype(np.uint8))
    #     # bbx_width = region_prop[0].bbox[3]-region_prop[0].bbox[1]
    #     # bbx_hight = region_prop[0].bbox[2]-region_prop[0].bbox[0]

    #     # self.height = int(math.sqrt(self.area/(bbx_width/bbx_hight)))
    #     # self.width = int((bbx_width/bbx_hight)*self.height)
    #     # self.matrix = np.zeros((_h, _w))
    #     # self.matrix[0:int(self.height),
    #     #             0:int(self.width)] = 1
    #     # self.center = [self.width/2, self.height/2]

    def transformed(self, transformation):
        """Transform the stone using the given transformation.

        :param transformation: Translation in x and y direction
        :type transformation: list
        :return: Translated stone
        :rtype: Stone object
        """
        # original_matirx = copy.deepcopy(self.matrix)
        # original_center = copy.deepcopy(self.center)

        # tform = transform.SimilarityTransform(scale=1, rotation=0,
        #                                       translation=[transformation[0], transformation[1]])
        # tranformed_matrix = transform.warp(original_matirx, tform.inverse)
        # tranformed_center = tform(original_center)[0]
        # 7 times faster than the above code
        shape = self.matrix.shape
        tranformed_matrix = np.zeros(shape)
        # print("transformation: ",transformation)
        # print("shape: ",shape)
        if transformation[0]>=0 and transformation[1]>=0 and transformation[0]<=shape[1] and transformation[1]<=shape[0]:
            tranformed_matrix[int(transformation[1]):, int(transformation[0]):] = self.matrix[:
                                                                                          shape[0]-int(transformation[1]), : shape[1]-int(transformation[0])]
            tranformed_center = [self.center[0] +
                             transformation[0], self.center[1]+transformation[1]]
            # tranformed_center = regionprops(tranformed_matrix.astype(np.uint8))[0].centroid
            # tranformed_center = [tranformed_center[1], tranformed_center[0]]
        else:
            tform = transform.SimilarityTransform(scale=1, rotation=0,translation=[transformation[0], transformation[1]])
            tranformed_matrix = transform.warp(self.matrix, tform.inverse)
            if len(regionprops(tranformed_matrix.astype(np.uint8)))==0:
                return None
            tranformed_center = regionprops(tranformed_matrix.astype(np.uint8))[0].centroid
            tranformed_center = [tranformed_center[1], tranformed_center[0]]

        # tranformed_center = [self.center[0] +
        #                      transformation[0], self.center[1]+transformation[1]]

        # # if transformed out of bound
        # if tranformed_matrix.sum() != self.matrix.sum():
        #     return None
        # if center out of bound
        if tranformed_center[0] < 0 or tranformed_center[0] > shape[1] or tranformed_center[1] < 0 or tranformed_center[1] > shape[0]:
            get_main_logger().debug(
                f'Transformation {transformation} apply to stone {self.id} but its center {tranformed_center} is out of bound')
            return None

        new_stone = Stone()
        new_stone.center = tranformed_center
        new_stone.matrix = tranformed_matrix
        new_stone.area = self.area
        new_stone.width = self.width
        new_stone.height = self.height
        new_stone.id = self.id
        new_stone.cluster = self.cluster
        new_stone.roundness = self.roundness
        new_stone.displacement_for_ka = self.displacement_for_ka
        new_stone.rotate_from_ori = self.rotate_from_ori
        return new_stone

    def plot(self):
        """Plot the stone
        """
        plt.imshow(np.flip(self.matrix, axis=0), cmap='gray')
        plt.scatter(
            self.center[0], (self.matrix.shape[0]-self.center[1]), c='r')
        plt.title(f'{self.id}')
        plt.axis('scaled')
        plt.show()

    def save(self, path):
        """Save the stone mask to file

        :param path: path to the file to be written
        :type path: str
        """
        plt.clf()
        plt.imshow(np.flip(self.matrix, axis=0), cmap='gray')
        plt.axis("off")
        plt.savefig(path)
    
    def save__scale(self,path,fig_length, fig_height):
        plt.clf()
        plt_matrix = np.flip(self.matrix[0:fig_height, 0:fig_length], axis=0)
        #plt_matrix[plt_matrix==0] = np.nan
        plt.imshow(plt_matrix, cmap='Greys')
        plt.axis("off")
        plt.savefig(path, dpi=600,transparent = True)

    def rotated_by_90(self):
        """
        Rotate the stone by 90 degree clockwise
        """
        # self.plot()
        # # rotate around(0, 0)
        # tform = transform.SimilarityTransform(scale=1, rotation=math.pi/2,
        #                                       translation=[0, 0])
        # tranformed_matrix = transform.warp(self.matrix, tform)
        # tranformed_center = tform.inverse(self.center)
        # plt.imshow(tranformed_matrix)
        # plt.scatter(tranformed_center[0][0], tranformed_center[0][1], c='r')
        # plt.show()
        # self.plot()

        ori_tranformed_matrix = np.flip(np.transpose(
            self.matrix), axis=1)
        #ori_tranformed_matrix = self.matrix
        # crop the flipped matrix
        true_stone = np.argwhere(ori_tranformed_matrix)
        top_left = true_stone.min(axis=0)
        bottom_right = true_stone.max(axis=0)
        cropped_matrix = ori_tranformed_matrix[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
        # plt.imshow(ori_tranformed_matrix)
        # plt.show()
        tranformed_matrix = np.zeros((self._h, self._w))
        tranformed_matrix[:cropped_matrix.shape[0],
                          :cropped_matrix.shape[1]] = cropped_matrix
        tranformed_center = [self.center[1], self.center[0]]
        # plt.imshow(tranformed_matrix)
        # plt.scatter(tranformed_center[0], tranformed_center[1], c='r')
        # plt.show()
        new_stone = Stone()
        new_stone.center = tranformed_center
        new_stone.matrix = tranformed_matrix
        new_stone.area = self.area
        new_stone.width = self.height
        new_stone.height = self.width
        new_stone._h = self._h
        new_stone._w = self._w
        new_stone.id = self.id
        new_stone.cluster = self.cluster
        new_stone.roundness = self.roundness
        new_stone.rotate_from_ori =self.rotate_from_ori+90
        # new_stone.plot()
        return new_stone

    def rotate_min_shape_factor(self):
        # '''
        # Function to rotate the stone to the position with the minimum shape factor
        # Step 1: rotate the stone to the center of the image
        # Step 2: iterate all angles with interval 1 degree, find the angle with the minimum shape factor
        # Step 3: move the minimal shape factor stone matrix to the left corner of image
        # Step 4: update stone information
        # '''

        # Step 1
        new_stone = self.transformed(
            [int(self.matrix.shape[1]/2)-self.center[0], int(self.matrix.shape[0]/2)-self.center[1]])
        stone_matrix = new_stone.matrix
        # Step 2
        min_angle = 0
        self.cal_shape_factor()
        min_sf = self.shape_factor
        for angle in range(-45, 45):
            rot_mat = cv2.getRotationMatrix2D(
                [int(self.matrix.shape[1]/2), int(self.matrix.shape[0]/2)], angle, 1.0)
            img_rotated = cv2.warpAffine(
                stone_matrix, rot_mat, stone_matrix.shape[1::-1], flags=cv2.WARP_FILL_OUTLIERS)
            img_rotated = cv2.medianBlur(img_rotated.astype(np.uint8), 3)
            # check if the stone too big for rotation
            if np.count_nonzero(img_rotated) / np.count_nonzero(stone_matrix)<0.8:
                print("Original size", np.count_nonzero(stone_matrix))
                print("Rotated size", np.count_nonzero(img_rotated))
                # plt.imshow(stone_matrix)
                # plt.title("Original matirx")
                # plt.show()
                # plt.imshow(img_rotated)
                # plt.title("Rotated matirx")
                # plt.show()
                return {"stone":None,"center":None,"angle":None,"success":False}
            try_stone = new_stone.transformed([0, 0])
            try_stone.matrix = img_rotated
            try_stone.cal_shape_factor()
            if try_stone.shape_factor < min_sf:
                min_angle = angle
                min_sf = try_stone.shape_factor
        #Angle is positive for anti-clockwise and negative for clockwise.
        rot_mat = cv2.getRotationMatrix2D(
            [int(self.matrix.shape[1]/2), int(self.matrix.shape[0]/2)], min_angle, 1.0)
        img_rotated = cv2.warpAffine(
            stone_matrix, rot_mat, stone_matrix.shape[1::-1], flags=cv2.WARP_FILL_OUTLIERS)
        #img_rotated = cv2.medianBlur(img_rotated.astype(np.uint8), 3)
        img_rotated = opening(img_rotated, square(3))
        #img_rotated = convex_hull_image(img_rotated,offset_coordinates=False)
        # Step 3
        true_stone = np.argwhere(img_rotated)
        top_left = true_stone.min(axis=0)
        bottom_right = true_stone.max(axis=0)
        cropped_matrix = img_rotated[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]

        new_stone.matrix = np.zeros((new_stone.matrix.shape))
        new_stone.matrix[:cropped_matrix.shape[0],
                         :cropped_matrix.shape[1]] = cropped_matrix

        new_stone.center = [new_stone.center[0] -
                            top_left[1], new_stone.center[1]-top_left[0]]
        # Step 4
        region_prop = regionprops(new_stone.matrix.astype(np.uint8))
        bbx_width = region_prop[0].bbox[3]-region_prop[0].bbox[1]
        bbx_hight = region_prop[0].bbox[2]-region_prop[0].bbox[0]
        new_stone.height = bbx_hight
        new_stone.width = bbx_width
        new_stone.id = self.id
        new_stone.cluster = self.cluster
        new_stone.roundness = self.roundness
        #new_stone.rot_angle+=min_angle
        get_main_logger().info(
            f"Rotated stone {self.id} to axis aligned position")
        return {"success":True,"stone":new_stone,"center":(int(self.matrix.shape[1]/2), int(self.matrix.shape[0]/2)),"angle":min_angle}

    def rotate_axis_align(self):
        """Rotate the stone so that the long axis of its equilvalent ellipse is aligned with the x-axis.

        :return: Rotated stone
        :rtype: Stone object
        """
        if get_plot_stone_id() == self.id:
            self.save(
                f"/home/qiwang/Projects/15_Stonepacker2D/img/stone{self.id}_start.png")
        # self.plot()
        # move stone to the center
        new_stone = self.transformed(
            [int(self.matrix.shape[1]/2)-self.center[0], int(self.matrix.shape[0]/2)-self.center[1]])
        # self.plot()
        # compute inertia direction of stone
        property = regionprops(new_stone.matrix.astype(np.uint8))
        angle = -property[0].orientation*180/math.pi+270
        # print(angle)
        rot_center = (property[0].centroid[1], property[0].centroid[0])
        rot_mat = cv2.getRotationMatrix2D(rot_center, angle, 1.0)
        img_rotated = cv2.warpAffine(
            new_stone.matrix, rot_mat, new_stone.matrix.shape[1::-1], flags=cv2.WARP_FILL_OUTLIERS)
        # plt.imshow(np.flip(img_rotated, axis=0))
        # plt.show()
        img_rotated = cv2.medianBlur(img_rotated.astype(np.uint8), 3)
        # plt.imshow(np.flip(img_rotated, axis=0))
        # plt.show()
        # crop the rotated matrix
        true_stone = np.argwhere(img_rotated)
        top_left = true_stone.min(axis=0)
        bottom_right = true_stone.max(axis=0)
        cropped_matrix = img_rotated[top_left[0]                                     :bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
        # print(cropped_matrix[5])
        new_stone.matrix = np.zeros((new_stone.matrix.shape))
        new_stone.matrix[:cropped_matrix.shape[0],
                         :cropped_matrix.shape[1]] = cropped_matrix

        new_stone.center = [new_stone.center[0] -
                            top_left[1], new_stone.center[1]-top_left[0]]
        # self.plot()
        region_prop = regionprops(new_stone.matrix.astype(np.uint8))
        bbx_width = region_prop[0].bbox[3]-region_prop[0].bbox[1]
        bbx_hight = region_prop[0].bbox[2]-region_prop[0].bbox[0]

        # new_stone.height = int(math.sqrt(new_stone.area/(bbx_width/bbx_hight)))
        # new_stone.width = int((bbx_width/bbx_hight)*new_stone.height)
        new_stone.height = bbx_hight
        new_stone.width = bbx_width
        new_stone.id = self.id
        new_stone.cluster = self.cluster
        new_stone.roundness = self.roundness
        get_main_logger().info(
            f"Rotated stone {self.id} to axis aligned position")
        if get_plot_stone_id() == self.id:
            self.save(
                f"/home/qiwang/Projects/15_Stonepacker2D/img/stone{self.id}_init.png")
        return {"stone":new_stone,"angle":angle}

    def cal_shape_factor(self):
        stone_matrix = np.zeros(self.matrix.shape)
        stone_matrix[1:, 1:] = self.matrix[:-1, :-1]
        properties = regionprops(stone_matrix.astype(np.uint8))
        # (min_row, min_col, max_row, max_col) half open interval
        bounding_pts = properties[0].bbox
        # print(bounding_pts)
        perimeter = properties[0].perimeter
        p_center = np.asarray(
            [properties[0].centroid[1], properties[0].centroid[0]])
        p0 = np.asarray([bounding_pts[3]-1, bounding_pts[2]-1])
        p1 = np.asarray([bounding_pts[1], bounding_pts[2]-1])
        p2 = np.asarray([bounding_pts[1], bounding_pts[0]])
        p3 = np.asarray([bounding_pts[3]-1, bounding_pts[0]])
        vec_l1 = (p0-p2)/np.linalg.norm(p0-p2)
        vec_l2 = (p1-p3)/np.linalg.norm(p1-p3)
        first_region_limit = np.dot(vec_l1, vec_l2)
        fourth_region_limit = np.dot(vec_l1, -vec_l2)

        diagonal_l1 = Line(Point(bounding_pts[1], bounding_pts[0], 0), Point(
            bounding_pts[3]-1, bounding_pts[2]-1, 0))
        # vec_l1 = Vector(Point(bounding_pts[1], bounding_pts[0], 0), Point(
        #     bounding_pts[3], bounding_pts[2], 0))
        diagonal_l2 = Line(Point(bounding_pts[1], bounding_pts[2]-1, 0), Point(
            bounding_pts[3]-1, bounding_pts[0], 0))
        # vec_l2 = Vector(Point(bounding_pts[1], bounding_pts[2], 0), Point(
        #     bounding_pts[3], bounding_pts[0], 0))
        # angle_l1_l2 = np.arctan
        # limit_anlge_l2 = vec_l1.angle(vec_l2*(-1))
        intersection_p = intersection(diagonal_l1, diagonal_l2)
        p_origion = np.asarray([intersection_p.x, intersection_p.y])
        # plt.clf()
        laplacian = cv2.Laplacian(stone_matrix,  cv2.CV_64F, ksize=3)
        contour_pts_rc = np.argwhere(laplacian < 0)
        # plt.imshow(laplacian, cmap='gray')
        groups = {0: [], 1: [], 2: [], 3: []}
        for i in range(contour_pts_rc.shape[0]):
            pt = np.asarray([contour_pts_rc[i][1], contour_pts_rc[i][0]])
            vec_pt = (pt-p_origion)/np.linalg.norm(pt-p_origion)
            cross_prod = np.cross(vec_l1, vec_pt)
            dot_prod = np.dot(vec_l1, vec_pt)
            if cross_prod > 0:
                if dot_prod > first_region_limit:
                    groups[0].append(pt)
                else:
                    groups[1].append(pt)
            else:
                if dot_prod > fourth_region_limit:
                    groups[3].append(pt)
                else:
                    groups[2].append(pt)

        # # plot the four groups
        # colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        #           for i in range(1000)]

        # for key, point_list in groups.items():
        #     for pt in point_list:
        #         plt.scatter(pt[0], pt[1], c=colors[key])
        # plt.scatter(p_center[0], p_center[1])
        # plt.show()
        perimeter = 2*(abs(p0[1]-p3[1])+abs(p0[0]-p1[0]))

        dk = np.zeros(len(groups[0]))
        for i, point in enumerate(groups[0]):
            dk[i] = abs(point[1]-p0[1])
        d0 = np.sqrt(np.sum(dk**2))/dk.shape[0]
        d0_normalized = 100*d0*abs(p0[0]-p1[0]) / \
            (abs(p0[1]-p_center[1])*perimeter)
        # print(dk)
        # print(np.sum(dk**2))
        # print(np.sqrt(np.sum(dk**2)))
        # print(d0)
        # print(d0_normalized)
        # print(abs(p0[1]-p_center[1]))

        dk = np.zeros(len(groups[1]))
        for i, point in enumerate(groups[1]):
            dk[i] = abs(point[0]-p1[0])
        d1 = np.sqrt(np.sum(dk**2))/dk.shape[0]
        d1_normalized = 100*d1*abs(p1[1]-p2[1]) / \
            (abs(p1[0]-p_center[0])*perimeter)
        # print(dk)
        # print(np.sum(dk**2))
        # print(np.sqrt(np.sum(dk**2)))
        # print(d1)
        # print(dk.shape)
        dk = np.zeros(len(groups[2]))
        for i, point in enumerate(groups[2]):
            dk[i] = abs(point[1]-p2[1])
        d2 = np.sqrt(np.sum(dk**2))/dk.shape[0]
        d2_normalized = 100*d2*abs(p2[0]-p3[0]) / \
            (abs(p2[1]-p_center[1])*perimeter)
        # print(dk)
        # print(np.sum(dk**2))
        # print(np.sqrt(np.sum(dk**2)))
        # print(d2)
        # print(dk.shape)
        dk = np.zeros(len(groups[3]))
        for i, point in enumerate(groups[3]):
            dk[i] = abs(point[0]-p3[0])
        d3 = np.sqrt(np.sum(dk**2))/dk.shape[0]
        d3_normalized = abs(100*d3*(p3[1]-p0[1]) /
                            ((p3[0]-p_center[0])*perimeter))
        # print(dk)
        # print(np.sum(dk**2))
        # print(np.sqrt(np.sum(dk**2)))
        # print(d3_normalized)
        # # print(dk.shape)
        # input(".......")
        self.shape_factor = (d0_normalized+d1_normalized +
                             d2_normalized+d3_normalized)/4

    def rotate_to_stable(self, final_base):
        other_stones = np.zeros(final_base.matrix.shape)
        base = final_base.matrix
        other_stones = base.copy()
        # find stones below
        indices = []
        stone_prop = regionprops(
            self.matrix.astype(np.uint8))
        low_bound = stone_prop[0].bbox[0]
        for index, brick in enumerate(final_base.placed_bricks):
            other_stone_prop = regionprops(brick.matrix.astype(np.uint8))
            if other_stone_prop[0].centroid[0] < low_bound:
                indices.append(index)
        base = np.zeros(final_base.matrix.shape)

        for index in indices:
            base += final_base.placed_bricks[index].matrix*(index+1)

        # find the contact by dilating 1 pixel
        img = self.matrix.astype(np.uint8)
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
        if stone_prop[0].centroid[1] > min_col and stone_prop[0].centroid[1] < max_col:
            # print("stable")
            rot_center = None
        elif stone_prop[0].centroid[1] <= min_col:  # rotate around row, min_col
            # rot_center = (row, min_col)
            rot_center = (min_col, row)
            mask = np.zeros(final_base.matrix.shape)
            mask[:, 0:int(stone_prop[0].centroid[1])] = 1
            rotate_clockwise = False
        elif stone_prop[0].centroid[1] >= max_col:
            # rot_center = (row, max_col)
            rot_center = (max_col, row)
            mask = np.zeros(final_base.matrix.shape)
            mask[:, int(stone_prop[0].centroid[1]):] = 1
            rotate_clockwise = True
        if rot_center is not None:
            if not rotate_clockwise:
                _min_angle = 0
                _max_angle = -90
                _delta_angle = -1
            else:
                _max_angle = 90
                _min_angle = 0
                _delta_angle = 1
            img = self.matrix.astype(np.uint8)
            for angle in range(_min_angle+_delta_angle, _max_angle, _delta_angle):
                # angle is positive for anti-clockwise, but image is flipped in terms of coordinates
                rot_mat = cv2.getRotationMatrix2D(rot_center, angle, 1.0)
                img_rotated = cv2.warpAffine(
                    img, rot_mat, img.shape[1::-1], flags=cv2.WARP_FILL_OUTLIERS)
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
                        intersection = img_rotated_dilated[0, :].reshape(
                            (1, -1))
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

                img_rotated = cv2.medianBlur(img_rotated.astype(np.uint8), 3)

                self.center = [
                    stone_prop[0].centroid[1], stone_prop[0].centroid[0]]
                self.matrix = img_rotated
                self.rot_center = rot_center
                self.rot_angle = angle

                return True
        else:
            return True

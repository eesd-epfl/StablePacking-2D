
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from Geometry3D import *
import numpy as np
from .transform import rotate_brick, check_stable, move_down,move_left,optimize_move
from ..utils.constant import get_cut_id,get_ignore_pixel_value,get_wedge_id, get_world_size, get_rotate_state, get_stabalize_method, get_base_id, get_dir
from .wedge import add_wedges
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
tab20_cm = cm.get_cmap('tab20')
newcolors = np.concatenate([tab20_cm(np.linspace(0, 1, 20))] * 13, axis=0)
white = np.array([255/256, 255/256, 255/256, 1])
newcolors[:1, :] = white
newcmp = ListedColormap(newcolors)


class Base():
    """Container of placed stones
    """

    def __init__(self):
        _w, _h = get_world_size()
        self.matrix = np.zeros((_h, _w))
        self.id_matrix = np.zeros((_h, _w))
        self.cluster_matrix = np.zeros((_h, _w))
        self.centers = []
        self.placed_bricks = []
        self._w = _w
        self._h = _h

        # wedge
        self.rock_centers = []
        self.placed_rocks = []
        self.rock_tops = []

        self.rock_right_bottoms = []
        self.rock_left_tops = []
        self.rock_right_tops = []
    def create_mirrored_base(self):
        """Create a mirrored base

        :return: Mirrored base
        :rtype: Base
        """
        mirrored_base = Base()
        mirrored_base.matrix = np.fliplr(self.matrix)
        mirrored_base.id_matrix = np.fliplr(self.id_matrix)
        mirrored_base.cluster_matrix = np.fliplr(self.cluster_matrix)
        for center in self.centers:
            mirrored_base.centers.append((self._w - center[0], center[1]))
        for brick in self.placed_bricks:
            mirrored_base.placed_bricks.append(brick.create_mirrored_stone(align_to_right = False))
        for rock in self.placed_rocks:
            mirrored_base.placed_rocks.append(rock.create_mirrored_stone(align_to_right = False))
        for rock_center in self.rock_centers:
            mirrored_base.rock_centers.append((self._w - rock_center[0], rock_center[1]))

        mirrored_base.rock_tops = self.rock_tops.copy()
        for rock_right_bottom in self.rock_right_bottoms:
            mirrored_base.rock_right_bottoms.append((self._w - rock_right_bottom[0], rock_right_bottom[1]))
        for rock_left_top in self.rock_right_tops:
            mirrored_base.rock_left_tops.append((self._w - rock_left_top[0], rock_left_top[1]))
        for rock_right_top in self.rock_left_tops:
            mirrored_base.rock_right_tops.append((self._w - rock_right_top[0], rock_right_top[1]))
        return mirrored_base

    def plot(self):
        # plot Matrix
        plt.imshow(np.flip(self.matrix, axis=0), cmap='gray')
        plt.axis('scaled')
        plt.show()

    def draw_last_action(self, plot_draw=False, save_draw=True, file_name=None):
        plt.close()
        fig, ax = plt.subplots()
        other_stones = np.where((self.matrix != 0)&(self.matrix!=get_ignore_pixel_value())&(self.matrix != len(self.placed_bricks)), 1, 0)
        last_action = np.where((self.matrix != 0)&(self.matrix!=get_ignore_pixel_value())&(self.matrix == len(self.placed_bricks)), 2, 0)
        ax.imshow(other_stones+last_action, cmap=newcmp,interpolation='none')
        plt.gca().invert_yaxis()
        plt.axis('off')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        if plot_draw:
            plt.show()
        if save_draw:
            plt.savefig(get_dir()+"img/"+file_name, dpi=300)
        plt.close()

    def draw_assembly(self, plot_draw=False, save_draw=True, file_name=None, sequence = False, number_type = 'cluster'):
        plt.close()
        fig, ax = plt.subplots()
        if number_type =='cluster':
            plt_matrix = np.where((self.matrix != 0)&(self.matrix!=get_ignore_pixel_value()), self.cluster_matrix+1, 0)
        elif number_type =='id':
            plt_matrix = np.where((self.matrix != 0)&(self.matrix!=get_ignore_pixel_value()), self.id_matrix+1, 0)
        
        ax.imshow(plt_matrix, cmap=newcmp,interpolation='none')
        text_kwargs = dict(ha='center', va='center',
                           fontsize=14, color='black')

        _text_size_threshold = min(self.matrix.shape[0],self.matrix.shape[1])/10       
        if not sequence:
            pass
            # for i, stone_i in enumerate(self.placed_rocks):
            #     if stone_i.height > _text_size_threshold and stone_i.width > _text_size_threshold:
            #         ax.text(self.rock_centers[i][0],
            #                 self.rock_centers[i][1], str(stone_i.id), **text_kwargs)
        else:
            for i, stone_i in enumerate(self.placed_rocks):
                if stone_i.height > _text_size_threshold and stone_i.width > _text_size_threshold:
                    ax.text(self.rock_centers[i][0],
                            self.rock_centers[i][1], str(i+1), **text_kwargs)
        plt.gca().invert_yaxis()
        plt.axis('off')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        if plot_draw:
            plt.show()
        if save_draw:
            plt.savefig(get_dir()+"img/"+file_name, dpi=600)
    

    def save(self, path):
        """Save the base mask to file

        :param path: path to the file to be written
        :type path: str
        """
        plt.clf()
        plt.imshow(np.flip(self.matrix, axis=0), cmap='gray')
        plt.axis("off")
        plt.savefig(path)

    def remove_brick(self,index):
        """Remove a stone from the base

        :param index: Index of the stone to be removed
        :type index: int
        """
        stone = self.placed_bricks[index]
        self.matrix-= stone.matrix*(index+1)
        self.id_matrix = np.where(self.matrix!=0,self.id_matrix,0)
        self.cluster_matrix = np.where(self.matrix!=0,self.cluster_matrix,0)
        self.placed_bricks.pop(index)
        self.centers.pop(index)
        
    def add_stone(self, stone, transformation,optimization_x_scale = [0,0],optimization_y_scale = [0,0]):
        """Place a stone with a given position

        :param stone: Stone 
        :type stone: Stone class object
        :param transformation: x and y of the position 
        :type transformation: list
        :return: Whether the placement was successful
        :rtype: bool
        """
        # transfor stone
        new_stone = stone.transformed(transformation)
        # add information to help debugging
        if new_stone is None:
            print("Warning: add a None type to base")
            print(
                f"Orginal stone center is {stone.center}, stone shape is {stone.width}*{stone.height}")
            print(f"Transformation is {transformation}")
            # img = plt.imshow(self.matrix)
            # plt.show()
            return False
        # update self matrix

        if get_rotate_state():
            new_stone.matrix = np.where(np.multiply(
                    new_stone.matrix, self.matrix) != 0, 0, new_stone.matrix)
            if np.count_nonzero(new_stone.matrix) == 0:
                print("ADDING a zero element to brick")
                return False
            if new_stone.id != get_wedge_id() and new_stone.id != get_base_id() and new_stone.id!= get_cut_id():
                if get_stabalize_method() == 'all':
                    new_stone = move_down(new_stone, self,optimization_y_scale = optimization_y_scale)
                    new_stone = move_left(new_stone, self,optimization_x_scale = optimization_x_scale)
                    new_stone = optimize_move(new_stone, self,optimization_x_scale = optimization_x_scale,optimization_y_scale = optimization_y_scale)
                self.matrix = np.add(
                    self.matrix, (len(self.centers)+1)*new_stone.matrix)
                self.id_matrix = np.add(
                    self.id_matrix, new_stone.id*new_stone.matrix)
                self.cluster_matrix = np.add(
                    self.cluster_matrix, new_stone.cluster*new_stone.matrix)
                self.centers.append(new_stone.center)
                self.placed_bricks.append(new_stone)
                # add to rock store
                self.rock_centers.append(new_stone.center)
                self.placed_rocks.append(new_stone)
                self.rock_tops.append(np.max(np.nonzero(new_stone.matrix)[0]))
                self.rock_right_bottoms.append([np.max(np.nonzero(new_stone.matrix)[
                                               1]), np.min(np.nonzero(new_stone.matrix)[0])])  # max col, min row
                self.rock_left_tops.append([np.min(np.nonzero(new_stone.matrix)[
                    1]), np.max(np.nonzero(new_stone.matrix)[0])])  # min col, max row
                self.rock_right_tops.append([np.max(np.nonzero(new_stone.matrix)[
                    1]), np.max(np.nonzero(new_stone.matrix)[0])])  # max col, max row

                current_index = int(len(self.centers)-1)
                if get_stabalize_method() == 'rotation':#DEFAULT: rotation first, if not stable, add wedge
                    rot_ok = rotate_brick(current_index, self)
                    #print(self.placed_bricks[current_index].rot_angle)
                    if not rot_ok:
                        _ = add_wedges([current_index], self)
                elif get_stabalize_method() == 'rotation_only':#DEFAULT: rotation first, if not stable, add wedge
                    rot_ok = rotate_brick(current_index, self)
                elif get_stabalize_method() == 'wedge':#wedge only
                    _ = add_wedges([current_index], self)
                elif get_stabalize_method() == 'all' or get_stabalize_method() == 'rot_wedge':# rotation and wedge sequentially
                    rot_ok = rotate_brick(current_index, self)
                    _ = add_wedges([current_index], self)
                elif get_stabalize_method() == 'none':# no stabalization
                    pass
            elif new_stone.id == get_cut_id():
                new_stone.matrix = np.where(np.multiply(
                    new_stone.matrix, self.matrix) != 0, 0, new_stone.matrix)
                if np.count_nonzero(new_stone.matrix) == 0:
                    print("ADDING a zero element to brick")
                    return True
                self.matrix = np.add(
                    self.matrix, (len(self.centers)+1)*new_stone.matrix)
                self.id_matrix = np.add(
                    self.id_matrix, new_stone.id*new_stone.matrix)
                self.cluster_matrix = np.add(
                    self.cluster_matrix, new_stone.cluster*new_stone.matrix)
                self.centers.append(new_stone.center)
                self.placed_bricks.append(new_stone)
                # stabalize
                current_index = int(len(self.centers)-1)
                _ = add_wedges([current_index], self)


            else:
                new_stone.matrix = np.where(np.multiply(
                    new_stone.matrix, self.matrix) != 0, 0, new_stone.matrix)
                if np.count_nonzero(new_stone.matrix) == 0:
                    print("ADDING a zero element to brick")
                    return True
                self.matrix = np.add(
                    self.matrix, (len(self.centers)+1)*new_stone.matrix)
                self.id_matrix = np.add(
                    self.id_matrix, new_stone.id*new_stone.matrix)
                self.cluster_matrix = np.add(
                    self.cluster_matrix, new_stone.cluster*new_stone.matrix)
                self.centers.append(new_stone.center)
                self.placed_bricks.append(new_stone)

        return True

    def get_filling(self):
        """Get the number of pixels filled

        :return: Number of filled pixels
        :rtype: int
        """
        return np.count_nonzero(self.matrix)

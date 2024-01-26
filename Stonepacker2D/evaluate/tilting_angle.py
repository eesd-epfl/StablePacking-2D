from scipy import ndimage
from ..utils.kine_2d import ContPoint, ContType
import numpy as np
from skimage.measure import regionprops
from ..utils.kine_2d import Element, update_elem_disp
import cv2
from ..utils.constant import get_ignore_pixel_value, get_world_size, get_mu, get_head_joint_state, get_kine_precision, get_dir
from ..utils.kine_2d import cal_Aglobal, solve_force_rigid, plot_elem_contp
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

tab20_cm = cm.get_cmap('tab20')
newcolors = np.concatenate([tab20_cm(np.linspace(0, 1, 20))] * 13, axis=0)
white = np.array([255/256, 255/256, 255/256, 1])
newcolors[:1, :] = white
newcmp = ListedColormap(newcolors)

# rot_cmap=cm.get_cmap('RdBu')
# rot_cmap[0] = white
class Parameter:

    # kinematics
    boundary = "tilting-table"
    stress = 30/100
    mu = 1.0
    fc = 0
    cohesion = 0
    density = 18
    calibrate_head_joint = False
    ignore_head_joint = False
    min_contp_distance = 1000
    offset = 1
    contact_detection_method = "uniform"
    ctname = "friction"


parameters = Parameter()
parameters_head = Parameter()
parameters_mortar = Parameter()


def plot_failure_mec(base_image, base_image_id, elems, LA, direction='right',plot_disp = 'euclidean',factor = 1e4):
    width = int(base_image.shape[1] * 1)
    height = int(base_image.shape[0] * 1)
    dim = (width, height)
    factor = factor
    moved_image = np.zeros(base_image.shape)
    # moved_image = cv2.resize(
    #     moved_image, dim, interpolation=cv2.INTER_NEAREST)
    max_rot = 0 
    max_x = 0
    max_y = 0
    max_euc = 0
    for key, element in elems.items():
        if abs(element.displacement[2]) > max_rot:
            max_rot = abs(element.displacement[2])
        if abs(element.displacement[0]) > max_x:
            max_x = abs(element.displacement[0])
        if abs(element.displacement[1]) > max_y:
            max_y = abs(element.displacement[1])
        if np.sqrt(abs(element.displacement[0])**2+abs(element.displacement[1])**2) > max_euc:
            max_euc = np.sqrt(abs(element.displacement[0])**2+abs(element.displacement[1])**2)
    for key, element in elems.items():
        if plot_disp == 'rotation':
            color_index = 0.01+(abs(element.displacement[2])/max_rot)
        elif plot_disp == 'x':
            color_index = 0.01+(abs(element.displacement[0])/max_x)
        elif plot_disp == 'y':
            color_index = 0.01+(abs(element.displacement[1])/max_y)
        elif plot_disp == 'euclidean':
            color_index = 0.01+(np.sqrt((abs(element.displacement[0])**2+abs(element.displacement[1])**2))/max_euc)
        else:
            color_index = 0.01
        #stone_pixels = np.where(base_image == element.id, base_image_id+1, 0)
        stone_pixels = np.where(base_image == element.id, color_index, 0)
        rot_center = element.center
        
        rot_mat = cv2.getRotationMatrix2D(
            (rot_center[0], rot_center[1]), element.displacement[2]*180/np.pi*factor, 1.0)
        rotated_stone_pixels = cv2.warpAffine(
            stone_pixels, rot_mat, stone_pixels.shape[1::-1], flags=cv2.WARP_FILL_OUTLIERS)
        T = np.float32([[1, 0, element.displacement[0]*factor],
                        [0, 1, element.displacement[1]*factor]])
        translated_stone_pixels = cv2.warpAffine(
            rotated_stone_pixels, T, base_image.shape[1::-1], flags=cv2.WARP_FILL_OUTLIERS)
        # translated_stone_pixels = cv2.resize(
        #     translated_stone_pixels, dim, interpolation=cv2.INTER_NEAREST)

        moved_image += translated_stone_pixels

    plt.clf()
    fig, ax = plt.subplots()
    #moved_image = np.where(moved_image == 1, 10, moved_image)
    moved_image = np.where(moved_image == 0, 0.5, moved_image)
    #moved_image[moved_image==0.5]=np.nan
    _plot = ax.imshow(moved_image, cmap='seismic',interpolation='none', vmin=0,vmax=1)
    # for key, value in elems.items():
    #     plt.scatter(value.center[0]*scale_+value.displacement[0]*factor*scale_,
    #                 value.center[1]*scale_+value.displacement[1]*factor*scale_, marker='^', c=color[value.id])

    # plt.axis('off')
    #plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    # text_kwargs = dict(ha='center', va='center', fontsize=20, color='C1')
    # for i, stone_i in enumerate(self.placed_rocks):
    #     plt.text(self.rock_centers[i][0],
    #             self.rock_centers[i][1], str(stone_i.id), **text_kwargs)

    # ax.text(0.5, 0.9, f"Limit load: {LA:.02f}\nForce direction {direction}", horizontalalignment='center',
    #         verticalalignment='center', transform=ax.transAxes, fontsize=16, color='black')
    plt.gca().invert_yaxis()
    plt.axis('off')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    cbar = plt.colorbar(_plot,cax = cax,orientation="horizontal")
    if plot_disp == 'rotation':
        cbar.set_label('element rotation/max rotation')
    elif plot_disp == 'x':
        cbar.set_label('element x displacement/max x displacement')
    elif plot_disp == 'y':
        cbar.set_label('element y displacement/max y displacement')
    elif plot_disp == 'euclidean':
        cbar.set_label('element displacement/max displacement')
    plt.savefig(get_dir()+"img/"+direction+str(LA)+'_mechanism.png', dpi=2400, transparent=True)


def evaluate_kine(final_base, save_failure=False, load='linear_acceleration'):
    """Evaluate the limit tilting angle of the placed stones

    :param final_base: The base to be evaluated
    :type final_base: Stonepacker2D.base.Base
    :return: The limit tilting angle of the placed stones, minimal of two direction
    :rtype: float
    """
    ignore_pixel_value = get_ignore_pixel_value()
    img = final_base.matrix.astype(np.uint8)
    # print(img.shape)
    img = np.where(img != ignore_pixel_value, img, 0)
    # print(img.shape)
    contps = pixel_to_mesh(img)
    elems = base_to_elements(final_base)
    contps = clear_contps(elems, contps)
    # ****************************************************************
    # ******** Tilting Table Load********************************
    # ****************************************************************
    # for element in elems.values():
    #     element.dl = [0, -element.mass, 0]
    #     element.ll = [element.mass, 0, 0]

    # ****************************************************************
    # ******** Horizontal Load on Last Element ********************************
    # ****************************************************************
    # last_element_id = element_id
    # total_mass = 0
    # for element in elems.values():
    #     element.dl = [0, -element.mass, 0]
    #     element.ll = [0, 0, 0]
    #     total_mass += element.mass
    # total_mass -= elems[last_element_id].mass
    # elems[last_element_id].dl = [0, -total_mass, 0]
    # elems[last_element_id].ll = [total_mass, 0, 0]

    # *****************************************************
    # ********** Linear Horizontal Load ***************************
    # *****************************************************
    _, wall_height = get_world_size()
    # for element in elems.values():
    #     element.dl = [0, -element.mass, 0]
    #     element.ll = [
    #         (1-math.cos(math.pi*element.center[1]/(2*wall_height)))*element.mass, 0, 0]
    if load == 'linear_acceleration':
        for element in elems.values():
            element.dl = [0, -element.mass, 0]
            element.ll = [
                (element.center[1]/wall_height)*element.mass, 0, 0]
    elif load == 'tilting_table':
        for element in elems.values():
            element.dl = [0, -element.mass, 0]
            element.ll = [element.mass, 0, 0]

    Aglobal = cal_Aglobal(elems, contps)
    #plot_elem_contp(elems, contps)
    one_direction_solution = solve_force_rigid(elems, contps, Aglobal)
    one_direction = one_direction_solution['limit_force']
    _plot_matrix = final_base.cluster_matrix.copy()
    if save_failure:
        update_elem_disp(elems, one_direction_solution['displacements'])
        plot_failure_mec(final_base.matrix, final_base.cluster_matrix,
                         elems, one_direction, direction='right',factor = 5e3)
        # plot_failure_mec(final_base.matrix, final_base.id_matrix,
        #                  elems, one_direction, direction='right')

    # ****************************************************************
    # ******** Tilting Table Load********************************
    # ***********************************************************
    # for element in elems.values():
    #     element.dl = [0, -element.mass, 0]
    #     element.ll = [-element.mass, 0, 0]
    # ****************************************************************
    # ******** Horizontal Load on Last Element********************************
    # ****************************************************************
    # elems[last_element_id].ll = [-total_mass, 0, 0]
    # *****************************************************
    # ********** Linear Horizontal Load ***************************
    # *****************************************************
    # for element in elems.values():
    #     element.dl = [0, -element.mass, 0]
    #     element.ll = [-(1-math.cos(math.pi*element.center[1] /
    #                     (2*wall_height)))*element.mass, 0, 0]
    if load == 'linear_acceleration':
        for element in elems.values():
            element.dl = [0, -element.mass, 0]
            element.ll = [-(element.center[1]/wall_height)*element.mass, 0, 0]
    elif load == 'tilting_table':
        for element in elems.values():
            element.dl = [0, -element.mass, 0]
            element.ll = [-element.mass, 0, 0]
    another_direction_solution = solve_force_rigid(elems, contps, Aglobal)
    another_direction = another_direction_solution['limit_force']
    if save_failure:
        update_elem_disp(elems, another_direction_solution['displacements'])
        # plot_failure_mec(final_base.matrix,
        #                  final_base.id_matrix, elems, another_direction, direction='left')
        plot_failure_mec(final_base.matrix, _plot_matrix,
                         elems, another_direction, direction='left',factor = 5e3)
        #print(one_direction, another_direction)
    # max_height = 0
    # for element in elems.values():
    #     if element.center[1] > max_height:
    #         max_height = element.center[1]
    #return 1
    return round(min(one_direction, another_direction), get_kine_precision())


def pixel_to_mesh(img,ignore_zero = True):
    contps = dict()
    maxPointID = 0
    parameters.mu = get_mu()

    _conttype = ContType("friction", parameters)
    parameters_head.mu = get_mu()
    if get_head_joint_state() == True:
        parameters_head.mu = 0
    _conttype_head = ContType("friction", parameters_head)

    # CONVOLVE: https://towardsdatascience.com/image-derivative-8a07a4118550
    # pixels whose value is different from pixel on the right

    normal = [-1, 0]
    tangent = [0, -1]
    Kx = np.array([[-1, 1, 0]])
    Fx_1 = ndimage.convolve(img, Kx)
    Kx = np.array([[1, -1, 0]])
    Fx_2 = ndimage.convolve(img, Kx)
    matrix_x_right = np.sum([Fx_1, Fx_2], axis=0)
    #matrix_x_right = np.multiply(img,matrix_x_right)
    index_x_right = np.argwhere(matrix_x_right > 0)
    coords = np.flip(np.sum([index_x_right, [[0, 0.5]]], axis=0), axis=1)
    # print(maxPointID)
    for i, coord in enumerate(coords):
        if ignore_zero:
            if img[index_x_right[i][0], index_x_right[i][1]] == 0 or img[index_x_right[i][0], min(index_x_right[i][1]+1, img.shape[1])] == 0:
                continue
        anta_id = img[index_x_right[i][0], min(
            index_x_right[i][1]+1, img.shape[1])]
        contps[maxPointID] = ContPoint(maxPointID, [
            coord[0], coord[1]], img[index_x_right[i][0], index_x_right[i][1]], anta_id, tangent, normal, _conttype_head)
        maxPointID += 1

    # pixels whose value is different from pixel on the left
    normal = [1, 0]
    tangent = [0, 1]
    Kx = np.array([[0, -1, 1]])
    Fx_1 = ndimage.convolve(img, Kx)
    Kx = np.array([[0, 1, -1]])
    Fx_2 = ndimage.convolve(img, Kx)
    matrix_x_left = np.sum([Fx_1, Fx_2], axis=0)
    matrix_x_left = np.multiply(img, matrix_x_left)
    index_x_left = np.argwhere(matrix_x_left > 0)
    coords = np.flip(np.sum([index_x_left, [[0, -0.5]]], axis=0), axis=1)
    # print(maxPointID)
    for i, coord in enumerate(coords):
        if ignore_zero:
            if img[index_x_left[i][0], index_x_left[i][1]] == 0 or img[index_x_left[i][0], max(index_x_left[i][1]-1, 0)] == 0:
                continue
        anta_id = img[index_x_left[i][0], max(index_x_left[i][1]-1, 0)]
        contps[maxPointID] = ContPoint(maxPointID, [
            coord[0], coord[1]], img[index_x_left[i][0], index_x_left[i][1]], anta_id, tangent, normal, _conttype_head)
        maxPointID += 1

    # pixels whose value is different from pixel below
    normal = [0, -1]
    tangent = [1, 0]
    Ky = np.array([[-1], [1], [0]])
    Fy_1 = ndimage.convolve(img, Ky)
    Ky = np.array([[1], [-1], [0]])
    Fy_2 = ndimage.convolve(img, Ky)
    matrix_y_below = np.sum([Fy_1, Fy_2], axis=0)
    matrix_y_below = np.multiply(img, matrix_y_below)
    index_y_below = np.argwhere(matrix_y_below > 0)
    coords = np.flip(np.sum([index_y_below, [[+0.5, 0]]], axis=0), axis=1)
    # print(maxPointID)
    for i, coord in enumerate(coords):
        if ignore_zero:
            if img[index_y_below[i][0], index_y_below[i][1]] == 0 or img[min(index_y_below[i][0]+1, img.shape[0]), index_y_below[i][1]] == 0:
                continue
        anta_id = img[min(index_y_below[i][0]+1, img.shape[0]),
                      index_y_below[i][1]]
        contps[maxPointID] = ContPoint(maxPointID, [
                                       coord[0], coord[1]], img[index_y_below[i][0], index_y_below[i][1]], anta_id, tangent, normal, _conttype)
        maxPointID += 1

    # pixels whose value is different from pixel above
    normal = [0, 1]
    tangent = [-1, 0]
    Ky = np.array([[0], [1], [-1]])
    Fy_1 = ndimage.convolve(img, Ky)
    Ky = np.array([[0], [-1], [1]])
    Fy_2 = ndimage.convolve(img, Ky)
    matrix_y_above = np.sum([Fy_1, Fy_2], axis=0)
    matrix_y_above = np.multiply(img, matrix_y_above)
    index_y_above = np.argwhere(matrix_y_above > 0)
    coords = np.flip(np.sum([index_y_above, [[-0.5, 0]]], axis=0), axis=1)
    # print(maxPointID)
    for i, coord in enumerate(coords):
        if ignore_zero:
            if img[index_y_above[i][0], index_y_above[i][1]] == 0 or img[max(index_y_above[i][0]-1, 0), index_y_above[i][1]] == 0:
                continue
        anta_id = img[max(index_y_above[i][0]-1, 0), index_y_above[i][1]]
        contps[maxPointID] = ContPoint(maxPointID, [
                                       coord[0], coord[1]], img[index_y_above[i][0], index_y_above[i][1]], anta_id, tangent, normal, _conttype)
        maxPointID += 1

    ####################################################################################################
    # print(len(contps))
    # print(len(contps))
    # for key, contp in contps.copy().items():
    #     n = np.asarray(contp.orientation[1])
    #     t = np.asarray(contp.orientation[0])
    #     for key_counter, contp_counter in contps.copy().items():
    #         n_counter = np.asarray(contp_counter.orientation[1])
    #         t_counter = np.asarray(contp_counter.orientation[0])
    #         # print(np.sum(np.multiply(n, n_counter)))
    #         # print(np.sum(np.multiply(n, n_counter)) == 0)
    #         # print(contp.coor == contp_counter.coor)
    #         # print(contp.cand == contp_counter.cand)
    #         # print(contp.anta == contp_counter.anta)
    #         if (contp.coor == contp_counter.coor):
    #             if np.sum(np.multiply(n, n_counter)) == 0:
    #                 if contp.cand == contp_counter.cand:
    #                     if contp.anta == contp_counter.anta:
    #                         # print("!!!!!!!!!!!!!!!!!")
    #                         if key in list(contps.keys()) and key_counter in list(contps.keys()):
    #                             del contps[key]
    #                             del contps[key_counter]
    #                             new_n = (n+n_counter) / \
    #                                 np.linalg.norm(n+n_counter)
    #                             _n_3d = np.asarray([new_n[0], new_n[1], 0])
    #                             _v = np.asarray([0, 0, 1])
    #                             _t = np.cross(_n_3d, _v)
    #                             new_t = _t[0:2]/np.linalg.norm(_t[0:2])
    #                             print(maxPointID, [
    #                                 contp.coor[0], contp.coor[1]], contp.cand, contp.anta, new_n.tolist(), new_t.tolist())

    #                             contps[maxPointID] = ContPoint(maxPointID, [
    #                                 contp.coor[0], contp.coor[1]], contp.cand, contp.anta, new_n.tolist(), new_t.tolist(), contp.cont_type)
    #                             # new_contps[(contp.coor[0], contp.coor[1], contp.cand, contp.anta)] = ContPoint(maxPointID, [
    #                             #     contp.coor[0], contp.coor[1]], contp.cand, contp.anta, ((n+n_counter)/1.414).to_list(), ((t+t_counter)/1.414).to_list(), contp.conttype)
    #                             # print(new_contps)
    #                             maxPointID += 1
    # print(len(contps))
    ####################################################################################################
    # for key, value in contps.items():
    #     if value.anta == 39:
    #         plt.imshow(np.flip(img, axis=0))
    #         plt.scatter(value.coor[0], value.coor[1])
    #         plt.show()
    return contps


def base_to_elements(final_base):
    elems = dict()
    for i, stone in enumerate(final_base.placed_bricks):
        element_id = i+1
        property = regionprops(stone.matrix.astype(np.uint8))
        if not property:
            print(stone.matrix)
            print(stone.id)
            if np.argwhere(stone.matrix).shape[0]==0:
                continue
            else:
                min_index = np.min(np.argwhere(stone.matrix), axis=0)
                max_index = np.max(np.argwhere(stone.matrix), axis=0)
                center = [0.5*(min_index[1]+max_index[1]),
                        0.5*(min_index[0]+max_index[0])]
        else:
            center = [property[0].centroid[1], property[0].centroid[0]]
        #edges = cv2.Canny(stone.matrix.astype(np.uint8), 0, 0)
        #vertices = np.flip(np.argwhere(edges > 0), axis=1)
        vertices = [[center[0]-stone.width/2, center[1]-stone.height/2], [center[0]-stone.width/2, center[1]+stone.height/2],
                    [center[0]+stone.width/2, center[1]+stone.height/2], [center[0]+stone.width/2, center[1]-stone.height/2]]
        if i == 0:
            elems[element_id] = Element(
                element_id, center, property[0].area, vertices, type='ground')
        else:
            elems[element_id] = Element(
                element_id, center, property[0].area, vertices, type='stone')

    return elems


def clear_contps(elems, contps):
    parameters.mu = get_mu()
    _conttype = ContType("friction", parameters)
    # make a loop up table for cand, anta -> contact point
    element_keys = list(elems.keys())
    contps_table = {(i, j): [] for i in element_keys
                    for j in element_keys}
    for key, value in contps.items():
        if value.anta not in element_keys or value.cand not in element_keys:
            print(value.coor)
            print(value.orientation)
        contps_table[(value.cand, value.anta)].append(value)

    cleaned_contps = dict()
    maxPointID = 0
    for key_cand, value_cand in elems.items():
        for key_anta, value_anta in elems.items():
            if value_cand.id != value_anta.id:
                all_contps = contps_table[(value_cand.id, value_anta.id)]
                if len(all_contps) != 0:
                    normals = np.zeros((len(all_contps), 2))
                    coordinates = np.zeros((len(all_contps), 2))
                    for i, pixel_point in enumerate(all_contps):
                        normals[i, :] = np.asarray(pixel_point.orientation[1])
                        coordinates[i, :] = np.asarray(pixel_point.coor)
                    averaged_normal = np.average(normals, axis=0)
                    if np.linalg.norm(averaged_normal) == 0:
                        continue
                    _n_3d = np.asarray(
                        [averaged_normal[0], averaged_normal[1], 0])
                    _v = np.asarray([0, 0, 1])
                    _t = np.cross(_n_3d, _v)
                    new_t = _t[0:2]/np.linalg.norm(_t[0:2])
                    new_n = averaged_normal/np.linalg.norm(averaged_normal)
                    # #check if n is vertical
                    # if new_n[1]==0:
                    #     parameters.mu = 0
                    # else:
                    #     parameters.mu = 0.58
                    # print(new_t)
                    #print(coordinates[:, 0])
                    # sort by projection to tangent direction
                    projections = np.matmul(
                        coordinates, np.asarray([[new_t[0]], [new_t[1]]]))
                    #print("cood: ", coordinates)
                    #print("projections: ", projections)
                    coordinate_sort_indices = np.argsort(
                        projections.reshape(1, -1))[0]
                    #print("order", coordinate_sort_indices)
                    # coordinate_sort_indices = np.lexsort(
                    #     (coordinates[:, 1], coordinates[:, 0]))
                    coordinate_end1 = coordinates[coordinate_sort_indices[0], :]
                    coordinate_end2 = coordinates[coordinate_sort_indices[-1], :]
                    if value_cand.type=='mortar' and value_anta.type == 'mortar':
                        _conttype = ContType("friction_fc_cohesion", parameters)
                    else:
                        _conttype = ContType("friction", parameters)
                    cleaned_contps[maxPointID] = ContPoint(maxPointID, [
                        coordinate_end1[0], coordinate_end1[1]], value_cand.id, value_anta.id, new_t.tolist(), new_n.tolist(), _conttype)
                    maxPointID += 1
                    cleaned_contps[maxPointID] = ContPoint(maxPointID, [
                        coordinate_end2[0], coordinate_end2[1]], value_cand.id, value_anta.id, new_t.tolist(), new_n.tolist(), _conttype)
                    maxPointID += 1
    return cleaned_contps

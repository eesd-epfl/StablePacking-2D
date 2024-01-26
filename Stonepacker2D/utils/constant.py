import numpy as np
import matplotlib
# https://github.com/wkentaro/labelme/issues/842
matplotlib.use('agg')
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
#np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
WORLD_WIDTH = 100
WORLD_HEIGHT = 100
EPS = -1e-5
MAX_EDGE_LENGTH = 50

# wall size
WALL_WIDTH = 100
WALL_HEIGHT = 100
OPTIMIZATION = "Bbox_Ocontour_Void"
# contour detection
DILATION = 1
EROSION = 1


# stacking algorithm
WEIGHT_FIRST_TO_ORIGIN_X = 0
ALLOW_ROTATE = True
DUMP_TO_FILE = True
SAVE_ROOT = "data/"
LOCAL_WIDTH = 0
INTER_WEIGHT = 0.1
WIDTH_WEIGHT = 0.1
STABALIZE_METHOD = 'rotation'
WEDGE_ID = 257
BASE_ID = 256
CUT_ID = 258
SELECTION_WEIGHTS = {'interlocking': 1, 'kinematic': 1,
                     'course_height': 1, 'global_density': 1, 'local_void':1,'local_void_center':1}
PLACEMENT_OPTIMIZATION = 'full_enumeration'
PARTIAL_ENUMERATION_RATIO = 0.5

# kinematics
MU = 0.58
DENSITY = 18
IGNORE_HEAD_JOINT = False
KINEMATIC_RESULT_DECIMAL = 1
IGNORE_VALUE = 255
# k steps ahead
K_STEPS = 0

# kinematics optimization
KINEMATIC_OPTIMIZATION = True
NUMBER_CPU = 8

#records
RECORD_TIME = True
RECORD_SWITCHS = {'RECORD_PLACEMENT_VALUE':True,'RECORD_PLACEMENT_IMG':True,'RECORD_SELECTION_VALUE':True}


def set_world_size(width, height):
    global WALL_WIDTH, WALL_HEIGHT
    WALL_WIDTH = width
    WALL_HEIGHT = height


def get_world_size():
    global WALL_WIDTH, WALL_HEIGHT
    return WALL_WIDTH, WALL_HEIGHT


def set_wall_size(width, height):
    global WORLD_WIDTH, WORLD_HEIGHT
    WORLD_WIDTH = width
    WORLD_HEIGHT = height


def get_wall_size():
    global WORLD_WIDTH, WORLD_HEIGHT
    return WORLD_WIDTH, WORLD_HEIGHT


def set_rotate_state(rotate):
    global ALLOW_ROTATE
    ALLOW_ROTATE = rotate


def get_rotate_state():
    global ALLOW_ROTATE
    return ALLOW_ROTATE


def set_dilation(dilation):
    global DILATION
    DILATION = dilation


def get_dilation():
    global DILATION
    return DILATION


def set_erosion(erosion):
    global EROSION
    EROSION = erosion


def get_erosion():
    global EROSION
    return EROSION


def set_ksteps(k):
    global K_STEPS
    K_STEPS = k


def get_ksteps():
    global K_STEPS
    return K_STEPS


def set_dump(state, dir):
    global DUMP_TO_FILE
    DUMP_TO_FILE = state
    global SAVE_ROOT
    SAVE_ROOT = dir


def get_dump():
    global DUMP_TO_FILE
    return DUMP_TO_FILE


def get_dir():
    global SAVE_ROOT
    return SAVE_ROOT


def set_mu(mu):
    global MU
    MU = mu


def get_mu():
    global MU
    return MU


def set_head_joint_state(state):
    global IGNORE_HEAD_JOINT
    IGNORE_HEAD_JOINT = state


def get_head_joint_state():
    global IGNORE_HEAD_JOINT
    return IGNORE_HEAD_JOINT


def get_local_width():
    global LOCAL_WIDTH
    return LOCAL_WIDTH


def set_local_width(width):
    global LOCAL_WIDTH
    LOCAL_WIDTH = width


def set_kine_precision(number_of_decimal):
    global KINEMATIC_RESULT_DECIMAL
    KINEMATIC_RESULT_DECIMAL = number_of_decimal


def get_kine_precision():
    global KINEMATIC_RESULT_DECIMAL
    return KINEMATIC_RESULT_DECIMAL


def set_number_cpu(number):
    global NUMBER_CPU
    NUMBER_CPU = number


def get_number_cpu():
    global NUMBER_CPU
    return NUMBER_CPU


def get_plot_placement():
    return 10


def get_plot_stone_id():
    return 255


def get_ignore_pixel_value():
    global IGNORE_VALUE
    return IGNORE_VALUE


def set_interlocking_weight(value):
    global INTER_WEIGHT
    INTER_WEIGHT = value


def get_interlocking_weight():
    global INTER_WEIGHT
    return INTER_WEIGHT


def get_stabalize_method():
    global STABALIZE_METHOD
    return STABALIZE_METHOD


def set_stabalize_method(value):
    global STABALIZE_METHOD
    STABALIZE_METHOD = value


def get_wedge_id():
    global WEDGE_ID
    return WEDGE_ID

def get_cut_id():
    global CUT_ID
    return CUT_ID


def get_base_id():
    global BASE_ID
    return BASE_ID


def get_width_weight():
    global WIDTH_WEIGHT
    return WIDTH_WEIGHT


def set_width_weight(value):
    global WIDTH_WEIGHT
    WIDTH_WEIGHT = value


def set_selection_weights(sw):
    global SELECTION_WEIGHTS
    #merge two dicts
    SELECTION_WEIGHTS = {**SELECTION_WEIGHTS, **sw}


def get_selection_weights():
    global SELECTION_WEIGHTS
    return SELECTION_WEIGHTS

def get_placement_optimization():
    global PLACEMENT_OPTIMIZATION
    return PLACEMENT_OPTIMIZATION
def set_placement_optimization(opt_algo):
    global PLACEMENT_OPTIMIZATION
    PLACEMENT_OPTIMIZATION = opt_algo

def get_partial_enumeration_ratio():
    global PARTIAL_ENUMERATION_RATIO
    return PARTIAL_ENUMERATION_RATIO

def set_partial_enumeration_ratio(ratio):
    global PARTIAL_ENUMERATION_RATIO
    PARTIAL_ENUMERATION_RATIO = ratio

def set_record_time(rt):
    global RECORD_TIME
    RECORD_TIME = rt

def get_record_time():
    global RECORD_TIME
    return RECORD_TIME

def set_record_detail(records):
    global RECORD_SWITCHS
    #merge two dicts
    RECORD_SWITCHS = {**RECORD_SWITCHS, **records}

def get_record_detail():
    global RECORD_SWITCHS
    return RECORD_SWITCHS
import cv2
def get_world_size_from_label_img(tif, scale=1):
    # crop the matrix
    matrix_from_img = np.flip(cv2.imread(tif)[:, :, 0], axis=0)
    # rescale image
    dim_0 = int(matrix_from_img.shape[1]*scale)
    dim_1 = int(matrix_from_img.shape[0]*scale)
    dim = (dim_0, dim_1)
    matrix_from_img = cv2.resize(
        matrix_from_img, dim, interpolation=cv2.INTER_NEAREST)
    # plt.imshow(matrix_from_img)
    # plt.show()
    true_stones = np.argwhere(matrix_from_img !=0)
    # print(true_stone)
    if true_stones.any():
        width = (true_stones.max(axis=0)-true_stones.min(axis=0))[1]
        height = (true_stones.max(axis=0)-true_stones.min(axis=0))[0]
        return width, height
    else:
        #raise errors
        raise Exception("No stones in the image")
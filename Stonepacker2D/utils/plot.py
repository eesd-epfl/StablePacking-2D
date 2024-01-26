import matplotlib.pyplot as plt
import numpy as np
from ..utils.constant import get_dir
import os
import cv2

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

tab20_cm = cm.get_cmap('tab20')
newcolors = np.concatenate([tab20_cm(np.linspace(0, 1, 256))] * 10, axis=0)
white = np.array([255/256, 255/256, 255/256, 1])
newcolors[:1, :] = white
newcmp = ListedColormap(newcolors)

_red = tab20_cm(6)
_bleu = tab20_cm(0)
_orange = tab20_cm(2)
newcolors = tab20_cm(np.linspace(0, 1, 4))
white = np.array([255/256, 255/256, 255/256, 1])
# red = np.array([255/256, 0, 0, 1.0])
# blue = np.array([0, 0, 255/256, 1.0])
# grey = np.array([200/256,200/256,200/256,1])
# yellow = np.array([255/256, 255/256, 0, 1.0])
newcolors[1:2, :]= _red
newcolors[2:3, :]= _bleu
newcolors[:,3] = 0.1
newcolors[3:4, :]= _orange
newcolors[3:4, 3]= 0.9
#newcolors[:,3] = 0.1
newcolors[:1, :] = white
cmp_wgbr = ListedColormap(newcolors)

def save_id_plot(matrix, filename, title='None'):
    plt.clf()
    fig, ax = plt.subplots()

    ax.imshow(matrix, cmap=newcmp)
    # plt.axis('off')
    text_kwargs = dict(ha='center', va='center',
                       fontsize=20, color='black')
    plt.gca().invert_yaxis()
    if title != "None":
        plt.text(0.5, 0.7, title, horizontalalignment='center',
                 verticalalignment='center', transform=ax.transAxes, fontsize=20, color='black')
    plt.savefig(get_dir()+"record/"+filename, transparent=True, dpi=300)


def save_matrix_plot(matrix, file_name,color = False):
    if not os.path.exists(get_dir()+"img/"):
        os.makedirs(get_dir()+"img/")
    plt.clf()
    if not color:
        plt.imshow(np.flip(matrix.astype(np.uint8), axis=0), cmap='Greys', interpolation='none')
    else:
        plt.imshow(np.flip(matrix.astype(np.uint8), axis=0),cmap = cmp_wgbr, interpolation='none',vmin = 0, vmax = 3)
    plt.axis("off")
    plt.savefig(get_dir()+"img/"+file_name, dpi=300)


def save_matrix(matrix, file_name):
    if not os.path.exists(get_dir()+"img/"):
        os.makedirs(get_dir()+"img/")
    cv2.imwrite(get_dir()+"img/"+file_name, matrix.astype(np.uint8))

def save_matrix_with_mask(matrix, mask,file_name,alpha = 0.1):
    if not os.path.exists(get_dir()+"img/"):
        os.makedirs(get_dir()+"img/")
    plt.clf()
    plt.imshow(np.flip(matrix.astype(np.uint8), axis=0), cmap='Greys', interpolation='none')
    plt.imshow(np.flip(mask.astype(np.uint8), axis=0),cmap = cmp_wgbr, alpha = 1.0,interpolation='none',vmin = 0, vmax = 3)
    plt.axis("off")
    plt.savefig(get_dir()+"img/"+file_name, dpi=300)

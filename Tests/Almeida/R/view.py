
import os
import pathlib

from Stonepacker2D import *

import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

import numpy as np


#----------------------------------------------------------------
# Set the seed

def seed_everything(seed=20):
    """"
    Seed everything.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything()
#----------------------------------------------------------------
# Set working directory

_root_dir = pathlib.Path(__file__).resolve().parent
_root_dir = os.path.abspath(_root_dir)

#----------------------------------------------------------------
# read pickle file
with open(_root_dir+"/result/base.pkl", 'rb') as f:
    final_base = pickle.load(f)



#-------------------------------PLOT
# Set the colormap---to highlight cutting
tab20_cm = cm.get_cmap('tab20')
_red = tab20_cm(6)
_bleu = tab20_cm(0)
newcolors = tab20_cm(np.linspace(0, 1, 4))
white = np.array([255/256, 255/256, 255/256, 1])
red = np.array([255/256, 0, 0, 0.7])
blue = np.array([0, 0, 255/256, 0.7])
grey = np.array([200/256,200/256,200/256,1])
newcolors[:,3] = 0.1
newcolors[:1, :] = white
newcolors[1:2, :]= grey
newcolors[2:3, :]= _bleu
newcolors[3:4, :]= _red
cmp_cut = ListedColormap(newcolors)
# --- view all stones -----
tab20_cm = cm.get_cmap('tab20')
color_260 = np.concatenate([tab20_cm(np.linspace(0, 1, 20))] * 13, axis=0)
white = np.array([255/256, 255/256, 255/256, 1])
color_260[:1, :] = white
for i_c,_c in enumerate(color_260[1:]):
    if np.allclose(_c,white,rtol=1e-05):
        color_260[i_c]=grey
cmp_260 = ListedColormap(color_260)

#-------------------------------
def remove_brick(base,index):
    """Remove a stone from the base

    :param index: Index of the stone to be removed
    :type index: int
    """
    stone = base.placed_bricks[index]
    base.matrix-= stone.matrix*(index+1)
    base.id_matrix = np.where(base.matrix!=0,base.id_matrix,0)
    base.cluster_matrix = np.where(base.matrix!=0,base.cluster_matrix,0)
    base.placed_bricks.pop(index)
    base.centers.pop(index)
#-------------------------------
# Plot the cut stones
fig, ax = plt.subplots()
original_stones = np.where((final_base.matrix != 0)&(final_base.matrix!=get_ignore_pixel_value())&(final_base.id_matrix!=get_cut_id())&(final_base.id_matrix!=get_wedge_id())&(final_base.id_matrix!=get_base_id()), 1, 0)
wedge_stones = np.where(final_base.id_matrix==get_wedge_id(),1,0)
cut_stones = np.where(final_base.id_matrix==get_cut_id(),1,0)
ax.imshow(original_stones+wedge_stones*2+cut_stones*3, cmap=cmp_cut,interpolation='none',vmin = 0,vmax = 3)

# Plot the action order
annotate_order = 0
for i, stone_i in enumerate(final_base.placed_bricks):
    if stone_i.id!=get_wedge_id() and stone_i.id!=get_base_id():
        if stone_i.height > 10 and stone_i.width > 10:
            text_kwargs = dict(ha='center', va='center',
                    fontsize=10, color='black')
        else:
            text_kwargs = dict(ha='center', va='center',
                        fontsize=min(stone_i.height,stone_i.width)-4, color='black')
        

        ax.text(stone_i.center[0],
                stone_i.center[1], str(annotate_order+1), **text_kwargs)
        annotate_order+=1
    elif stone_i.id==get_wedge_id():
        text_kwargs = dict(ha='center', va='center',
                        fontsize=min(stone_i.height,stone_i.width)-4, color='black')
        ax.text(stone_i.center[0],
                stone_i.center[1], f"{annotate_order}W", **text_kwargs)
    else:
        continue

plt.gca().invert_yaxis()
plt.axis('off')
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
plt.savefig(_root_dir+"/result/img/cut_wedge_stone.pdf", dpi=300)
#-------------------------------
# Plot the wedge stones
fig, ax = plt.subplots()
original_stones = np.where((final_base.matrix != 0)&(final_base.matrix!=get_ignore_pixel_value())&(final_base.id_matrix!=get_wedge_id())&(final_base.id_matrix!=get_base_id()), 1, 0)
wedge_stones = np.where(final_base.id_matrix==get_wedge_id(),1,0)
ax.imshow(original_stones+wedge_stones*2, cmap=cmp_cut,interpolation='none',vmin = 0,vmax = 3)

# Plot the action order
annotate_order = 0
for i, stone_i in enumerate(final_base.placed_bricks):
    if stone_i.id!=get_wedge_id() and stone_i.id!=get_base_id():
        if stone_i.height > 10 and stone_i.width > 10:
            text_kwargs = dict(ha='center', va='center',
                    fontsize=10, color='black')
        else:
            text_kwargs = dict(ha='center', va='center',
                        fontsize=min(stone_i.height,stone_i.width)-4, color='black')
        

        ax.text(stone_i.center[0],
                stone_i.center[1], str(annotate_order+1), **text_kwargs)
        annotate_order+=1
    elif stone_i.id==get_wedge_id():
        text_kwargs = dict(ha='center', va='center',
                        fontsize=min(stone_i.height,stone_i.width)-4, color='black')
        ax.text(stone_i.center[0],
                stone_i.center[1], f"{annotate_order}W", **text_kwargs)
    else:
        continue

plt.gca().invert_yaxis()
plt.axis('off')
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
plt.savefig(_root_dir+"/result/img/wedge_stone.pdf", dpi=300)
#-------------------------------
# Plot the all stones
plt.close()
fig, ax = plt.subplots()
normal_stones_cluster = np.where((final_base.matrix != 0)&(final_base.matrix!=get_ignore_pixel_value()), (final_base.cluster_matrix+1), 0)
wedge_stones = np.where(final_base.id_matrix==get_wedge_id(),1,0)
cut_stones = np.where(final_base.id_matrix==get_cut_id(),1,0)
ax.imshow(wedge_stones*1.5+cut_stones*2.5+normal_stones_cluster, cmap=cmp_260,interpolation='none')
# Plot the action order
annotate_order = 0
for i, stone_i in enumerate(final_base.placed_bricks):
    if stone_i.id!=get_wedge_id() and stone_i.id!=get_base_id() and stone_i.id!=get_cut_id():
        if stone_i.height > 10 and stone_i.width > 10:
            text_kwargs = dict(ha='center', va='center',
                    fontsize=10, color='black')
        else:
            text_kwargs = dict(ha='center', va='center',
                        fontsize=min(stone_i.height,stone_i.width)-4, color='black')
        

        ax.text(stone_i.center[0],
                stone_i.center[1], str(annotate_order+1), **text_kwargs)
        annotate_order+=1
    elif stone_i.id==get_wedge_id():
        text_kwargs = dict(ha='center', va='center',
                        fontsize=min(stone_i.height,stone_i.width)-4, color='black')
        ax.text(stone_i.center[0],
                stone_i.center[1], f"{annotate_order}W", **text_kwargs)
    elif stone_i.id== get_cut_id():
        if stone_i.height > 10 and stone_i.width > 10:
            text_kwargs = dict(ha='center', va='center',
                    fontsize=10, color='black')
        else:
            text_kwargs = dict(ha='center', va='center',
                        fontsize=min(stone_i.height,stone_i.width)-4, color='black')
        

        ax.text(stone_i.center[0],
                stone_i.center[1], f"{annotate_order+1}S", **text_kwargs)
        annotate_order+=1
    else:
        continue

plt.gca().invert_yaxis()
plt.axis('off')
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
plt.savefig(_root_dir+"/result/img/wall_stone_cluster.pdf", dpi=300)
#-------------------------------
#Plot -1 stones
remove_brick(final_base,len(final_base.placed_bricks)-1)
plt.close()
fig, ax = plt.subplots()
normal_stones_cluster = np.where((final_base.matrix != 0)&(final_base.matrix!=get_ignore_pixel_value()), (final_base.cluster_matrix+1), 0)
wedge_stones = np.where(final_base.id_matrix==get_wedge_id(),1,0)
cut_stones = np.where(final_base.id_matrix==get_cut_id(),1,0)
ax.imshow(wedge_stones*1.5+cut_stones*2.5+normal_stones_cluster, cmap=cmp_260,interpolation='none')
# Plot the action order
annotate_order = 0
for i, stone_i in enumerate(final_base.placed_bricks):
    if stone_i.id!=get_wedge_id() and stone_i.id!=get_base_id() and stone_i.id!=get_cut_id():
        if stone_i.height > 10 and stone_i.width > 10:
            text_kwargs = dict(ha='center', va='center',
                    fontsize=10, color='black')
        else:
            text_kwargs = dict(ha='center', va='center',
                        fontsize=min(stone_i.height,stone_i.width)-4, color='black')
        

        ax.text(stone_i.center[0],
                stone_i.center[1], str(annotate_order+1), **text_kwargs)
        annotate_order+=1
    elif stone_i.id==get_wedge_id():
        text_kwargs = dict(ha='center', va='center',
                        fontsize=min(stone_i.height,stone_i.width)-4, color='black')
        ax.text(stone_i.center[0],
                stone_i.center[1], f"{annotate_order}W", **text_kwargs)
    elif stone_i.id== get_cut_id():
        if stone_i.height > 10 and stone_i.width > 10:
            text_kwargs = dict(ha='center', va='center',
                    fontsize=10, color='black')
        else:
            text_kwargs = dict(ha='center', va='center',
                        fontsize=min(stone_i.height,stone_i.width)-4, color='black')
        

        ax.text(stone_i.center[0],
                stone_i.center[1], f"{annotate_order+1}S", **text_kwargs)
        annotate_order+=1
    else:
        continue

plt.gca().invert_yaxis()
plt.axis('off')
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
plt.savefig(_root_dir+"/result/img/wall_stone-1_cluster.pdf", dpi=300)

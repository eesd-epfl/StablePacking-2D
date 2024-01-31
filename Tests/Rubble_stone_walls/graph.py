import numpy
from scipy.ndimage import label
import networkx as nx
import pickle
import json
import os
import math
import cv2
from cv2 import repeat
import matplotlib.pyplot as plt
import pathlib
import math

import numpy as np

from scipy import ndimage
import numpy as np
from skimage.measure import regionprops
from Stonepacker2D import *
import matplotlib


def build_graph(img, erosion=True):
    if erosion:
        kernel_erosion = np.ones((4, 4), np.uint8)
        image = cv2.erode(
            img, kernel_erosion, iterations=1)
    else:
        image = img
    #image = img
    # x, y = np.meshgrid(np.linspace(-np.pi/2, np.pi/2, 30),
    #                    np.linspace(-np.pi/2, np.pi/2, 30))
    # image = (np.sin(x**2+y**2)[1:-1, 1:-2] > 0.9).astype(int)
    # print(image)

    # Mortar edges
    # CONSTRUCTION OF HORIZONTAL EDGES
    # horizontal edge start positions
    hx, hy = np.where((image[1:, :] == 0) & (image[:-1, :] == 0))
    h_units = np.array([hx, hy]).T
    h_starts = [tuple(n) for n in h_units]
    # end positions = start positions shifted by vector (1,0)
    h_ends = [tuple(n) for n in h_units + (1, 0)]
    h_weights = [1]*len(h_ends)
    horizontal_edges = zip(h_starts, h_ends, h_weights)

    # CONSTRUCTION OF VERTICAL EDGES
    # vertical edge start positions
    vx, vy = np.where((image[:, 1:] == 0) & (image[:, :-1] == 0))
    v_units = np.array([vx, vy]).T
    v_starts = [tuple(n) for n in v_units]
    # end positions = start positions shifted by vector (0,1)
    v_ends = [tuple(n) for n in v_units + (0, 1)]
    v_weights = [1]*len(h_ends)
    vertical_edges = zip(v_starts, v_ends, v_weights)

    # Construction of sloped edges
    # to right bottom
    sx, sy = np.where((image[1:, 1:] == 0) & (image[:-1, :-1] == 0))
    s_units = np.array([sx, sy]).T
    s_starts = [tuple(n) for n in s_units]
    # end positions = start positions shifted by vector (0,1)
    s_ends = [tuple(n) for n in s_units + (1, 1)]
    s_weights = [1.44]*len(h_ends)
    s_r_edges = zip(s_starts, s_ends, s_weights)

    # Construction of sloped edges
    # to left bottom
    sx, sy = np.where((image[:-2, 1:] == 0) & (image[1:-1, :-1] == 0))
    s_units = np.array([sx+1, sy]).T
    s_starts = [tuple(n) for n in s_units]
    # end positions = start positions shifted by vector (0,1)
    s_ends = [tuple(n) for n in s_units + (-1, 1)]
    s_weights = [1.44]*len(h_ends)
    s_l_edges = zip(s_starts, s_ends, s_weights)

    G = nx.Graph()
    G.add_weighted_edges_from(horizontal_edges)
    G.add_weighted_edges_from(vertical_edges)
    G.add_weighted_edges_from(s_r_edges)
    G.add_weighted_edges_from(s_l_edges)

    # Interface edges
    for u, v, e in G.edges(data=True):
        if len(np.argwhere(image[max(0, u[0]-1):u[0]+2, max(0, u[1]-1):u[1]+2] != 0)) != 0 and len(np.argwhere(image[max(0, v[0]-1):v[0]+2, max(0, v[1]-1):v[1]+2] != 0)) != 0:
            e['weight'] *= 0.3

    return G


# pos = dict(zip(G.nodes(), G.nodes()))  # map node names to coordinates
# nx.draw_networkx(G, pos, with_labels=False, node_size=0)
# labels = nx.get_edge_attributes(G, 'weight')
# #labels = {node: f'({node[0]},{node[1]})' for node in G.nodes()}
# nx.draw_networkx_edge_labels(G, pos, labels, font_size=6, font_family='serif',
#                              font_weight='bold', bbox=dict(fc='lightblue', ec="black", boxstyle="round", lw=1))
# plt.show()
# nx.draw_networkx(G, pos, with_labels=False, node_size=0)
# labels = {node: f'({node[0]},{node[1]})' for node in G.nodes()}
# nx.draw_networkx_labels(G, pos, labels, font_size=6, font_family='serif',
#                         font_weight='bold', bbox=dict(fc='lightblue', ec="black", boxstyle="round", lw=1))
# plt.show()
def plot_LMT(image, paths, LMTs, _root_dir):
    plt.clf()
    image[image==0] = np.nan
    plt.imshow(image, cmap='gray')
    for i, path in enumerate(paths):
        centers_r = []
        centers_c = []
        for node in path:
            centers_r.append(node[0])
            centers_c.append(node[1])
        plt.plot(centers_c, centers_r,
                 '-o', c='r', label=f"LMT: {round(LMTs[i],3)}")
        leg = plt.legend(fontsize=10, bbox_to_anchor=(
            0, 0.9), ncol=2, loc="lower left")
        leg.get_frame().set_linewidth(0.0)
        leg.get_frame().set_alpha(0.0)
    plt.axis('off')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(_root_dir+"/data/"+
                "LMT.png", dpi=600,transparent = True, bbox_inches='tight')
    plt.show()

def plot_FAV(image, paths, FAV_100, _root_dir):
    plt.clf()
    image[image==0] = np.nan
    plt.imshow(image, cmap='gray')
    for i, path in enumerate(paths):
        centers_r = []
        centers_c = []
        for node in path:
            centers_r.append(node[0])
            centers_c.append(node[1])
        plt.plot(centers_c, centers_r,
                 '-o', c='r',ms = 2)
    # plt.text(1, -2, f" FAV(%){round(FAV_100,2)}")
    plt.axis('off')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(_root_dir+"/data/"+
                "FAV.png", dpi=600,transparent = True, bbox_inches='tight')
    plt.show()
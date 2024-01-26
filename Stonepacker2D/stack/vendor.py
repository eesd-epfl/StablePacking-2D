from copy import deepcopy
from xml.sax.handler import feature_external_ges
import numpy as np
from ..utils.logger import get_main_logger
from ..utils.constant import get_dir,get_record_detail
from skimage.measure import regionprops, find_contours
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, MeanShift, estimate_bandwidth,AgglomerativeClustering
import cv2
import matplotlib.pyplot as plt
import random
import pandas as pd
np.random.seed(1307)
plt.style.use('science')
from matplotlib import cm
from matplotlib.colors import ListedColormap
tab20_cm = cm.get_cmap('tab20')
newcolors = np.concatenate([tab20_cm(np.linspace(0, 1, 20))] * 13, axis=0)
white = np.array([255/256, 255/256, 255/256, 1])
newcolors[:1, :] = white
newcmp = ListedColormap(newcolors)

class Vendor():
    def __init__(self, stones,renew_id = True):
        self.stones = dict()
        self.max_stone_id = len(self.stones.values())
        for stone in stones:
            if renew_id:
                self.stones[self.max_stone_id] = stone
                self.stones[self.max_stone_id].id = int(self.max_stone_id)
                self.max_stone_id += 1
            else:
                self.stones[stone.id] = stone
                #print(stone.id)
                if self.max_stone_id<stone.id:
                    self.max_stone_id = stone.id

        self.labels = dict()

    def reorder_dict(self):
        shuffeld_stones = dict()
        original_keys = list(self.stones.keys())
        #shuffle the keys
        np.random.shuffle(original_keys)
        for key in original_keys:
            shuffeld_stones[key] = self.stones[key]
        self.stones = shuffeld_stones
    
    def get_new_stone_id(self):
        self.max_stone_id += 1
        return self.max_stone_id
    
    def get_max_stone_size(self):
        #check if the dict is empty
        if len(self.stones.keys()) == 0:
            return 0
        max_size = 0
        for stone in self.stones.values():
            if stone.area > max_size:
                max_size = stone.area
        return max_size
    def get_min_stone_size(self):
        #check if the dict is empty
        if len(self.stones.keys()) == 0:
            return np.inf
        min_size = np.inf
        for stone in self.stones.values():
            if stone.area <min_size:
                min_size = stone.area
        return min_size
    
    def get_max_width(self):
        if len(self.stones.keys()) == 0:
            return np.inf
        max_width = 0
        for stone in self.stones.values():
            if stone.width > max_width:
                max_width = stone.width
        return max_width

    def get_random_stones(self, nb_stones):
        stone_keys = list(self.stones.keys())
        if nb_stones > len(stone_keys):
            get_main_logger().warning("Not enough stones from the vendor")
            selected_stone_ids = stone_keys
        else:
            selected_stone_ids = np.random.choice(
                stone_keys, size=nb_stones, replace=False)
        selected_stones = []
        for id in selected_stone_ids:
            selected_stones.append(self.stones[id])
            del self.stones[id]
        return selected_stones

    def return_stones(self, stones):
        for stone in stones:
            self.stones[stone.id] = stone

    def get_number_stones(self):
        return len(self.stones.keys())

    def cluster_stones(self, nb_clusters=8,plot_figure = True,plot_cluster_txt = False):
        # stone shape
        # Rec 1: bounding box
        # Rec 2: scaled 1
        # Rec 3: equilvalent area and perimeter
        # two lines->4 regions
        features = np.zeros((len(self.stones.keys()), 6))
        for i, value in enumerate(self.stones.values()):
            #print(value.width/value.height, value.height, value.shape_factor)
            # features[i] = np.asarray(
            #     [value.width,value.height,value.width*value.height, value.height/value.width,value.roundness])
            #print(value.id)
            features[i] = np.asarray(
                [value.width,value.height,value.area, value.height/value.width,value.roundness,value.id])
           
        df = pd.DataFrame(features, columns=['width', 'height','area','hw_ratio','roundness','id'])

        # # scaler = StandardScaler()
        # # scaled_features = scaler.fit_transform(features)
        # kmeans = KMeans(init='random', n_clusters=nb_clusters,
        #                 n_init=10, max_iter=300)
        # sc = SpectralClustering(
        #     nb_clusters, affinity='rbf', n_init=100, assign_labels='discretize')
        # # _____________________________________________________________________ DBSCAN
        # db = DBSCAN(eps=5).fit(features)
        # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        # core_samples_mask[db.core_sample_indices_] = True
        # labels = db.labels_

        # # Number of clusters in labels, ignoring noise if present.
        # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        # n_noise_ = list(labels).count(-1)
        # # Black removed and is used for noise instead.
        # plt.clf()
        # unique_labels = set(labels)
        # colors = [plt.cm.Spectral(each)
        #           for each in np.linspace(0, 1, len(unique_labels))]
        # for k, col in zip(unique_labels, colors):
        #     if k == -1:
        #         # Black used for noise.
        #         col = [0, 0, 0, 1]

        #     class_member_mask = labels == k

        #     xy = features[class_member_mask & core_samples_mask]
        #     plt.plot(
        #         xy[:, 0],
        #         xy[:, 1],
        #         "o",
        #         markerfacecolor=tuple(col),
        #         markeredgecolor="k",
        #         markersize=14,
        #     )

        #     xy = features[class_member_mask & ~core_samples_mask]
        #     plt.plot(
        #         xy[:, 0],
        #         xy[:, 1],
        #         "o",
        #         markerfacecolor=tuple(col),
        #         markeredgecolor="k",
        #         markersize=6,
        #     )

        # plt.title("Estimated number of clusters: %d" % n_clusters_)
        # plt.show()
        # # ____________________________________________________________________-DBSCAN END
        # ______________________________________________________________________MEAN SHIFT
        # The following bandwidth can be automatically detected using
        # https://stackoverflow.com/questions/28335070/how-to-choose-appropriate-quantile-value-while-estimating-bandwidth-in-meanshift
        X = features[:,[2,4]].copy()
        X[:,0] = X[:,0]/X[:,0].max()

        if get_record_detail()['RECORD_PLACEMENT_IMG']:
            for i, value in enumerate(self.stones.values()):
                value.save__scale(get_dir()+f"img/stone_{value.id}_{X[i,0]}_{X[i,1]}.png",int(features[:,0].max()),int(features[:,1].max()))
        # transformer = RobustScaler().fit(X)
        # X = transformer.transform(X)
        
        bandwidth = estimate_bandwidth(X, quantile=0.1)
        if bandwidth <= 0:
            bandwidth = 1

        #ms = MeanShift(bandwidth=bandwidth, bin_seeding=False)
        #ms = DBSCAN(eps = 0.07, min_samples=1,leaf_size = 3)
        ms = AgglomerativeClustering(n_clusters = None, distance_threshold = 0.2,linkage = 'average')
        ms.fit(X)
        labels = ms.labels_
        #cluster_centers = ms.cluster_centers_
        # assight label to stones
        for i, value in enumerate(self.stones.values()):
            value.cluster = labels[i]
            try:
                self.labels[labels[i]].append(value.id)
            except:
                self.labels[labels[i]] = [value.id]

        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)

        print("number of estimated clusters : %d" % n_clusters_)
        if plot_figure:
            from itertools import cycle

            plt.figure(1)
            plt.clf()

            colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
            for i in range(n_clusters_):
                color = newcmp((i+1)/n_clusters_)
                plt.plot(X[labels == i, 0], X[labels == i, 1], c = color,linestyle = 'None',marker=".",markersize=10)
                if plot_cluster_txt:
                    plt.text(X[labels == i, 0].mean(), X[labels == i, 1].mean(),str(i),size=20, color=color)
            # for k, col in zip(range(n_clusters_), colors):
            #     my_members = labels == k
            #     #cluster_center = cluster_centers[k]
            #     plt.plot(X[my_members, 0],
            #             X[my_members, 1], col + ".",markersize=10)
            #     # plt.plot(
            #     #     cluster_center[0],
            #     #     cluster_center[1]*cluster_center[0],
            #     #     "o",
            #     #     markerfacecolor=col,
            #     #     markeredgecolor="k",
            #     #     markersize=14,
            #     # )
            #plt.title("Estimated number of clusters: %d" % n_clusters_)
            plt.ylabel("Eccentricity")
            plt.xlabel("Size")
            #plt.axis('equal')
            #set x limit
            plt.xlim(0,1.05)
            plt.ylim(0,1.05)
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
            plt.savefig(get_dir()+f"img/cluster_{plot_cluster_txt}.pdf", dpi=600,transparent=True)
        # plt.show()
        # _______________________________________________________________MEAN SHIFT END
        # # kmeans.fit(scaled_features)
        # df['cluster'] = db.fit_predict(df[['width', 'height']])
        # # get centroids
        # #centroids = sc.cluster_centers_
        # #cen_x = [i[0] for i in centroids]
        # #cen_y = [i[1] for i in centroids]
        # # add to df
        # # df['cen_x'] = df.cluster.map({0: cen_x[0], 1: cen_x[1], 2: cen_x[2], 3: cen_x[3], 4: cen_x[4], 5: cen_x[5],
        # #                              6: cen_x[6], 7: cen_x[7]})
        # # df['cen_y'] = df.cluster.map(
        # #    {0: cen_y[0], 1: cen_y[1], 2: cen_y[2], 3: cen_y[3], 4: cen_y[4], 5: cen_y[5],
        # #     6: cen_y[6], 7: cen_y[7]})  # define and map colors
        # colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        #           for i in range(1000)]
        # df['c'] = df.cluster.map({0: colors[0], 1: colors[1], 2: colors[2],
        #                          3: colors[3], 4: colors[4], 5: colors[5], 6: colors[6], 7: colors[7]})
        # plt.clf()
        # plt.scatter(df.width, df.height, c=df.c, alpha=0.6, s=10)
        # plt.show()

        for i in range(n_clusters_):
            #self.labels[i] = np.argwhere(ms.labels_ == i)[:, -1]
            self.labels[i] = np.asarray(self.labels[i])
            # print(i)
            # for j in self.labels[i]:
            #     self.stones[j].plot()
        return n_clusters_

    # def sort_clusters(self):
    #     for key, value in self.labels.copy().items():
    #         sorted_indexs = sorted(value, key=lambda i: -self.stones[i].width)
    #         self.labels[key] = sorted_indexs

        # for i in self.labels[0]:
        #     self.stones[i].plot()

    def get_all_stones(self):
        selected_stones = []
        for key, value in self.stones.copy().items():
            selected_stones.append(value)
            del self.stones[key]
        return selected_stones
    
    def remove_stone_by_id(self,ids):
        add_stones = []
        del_ids = []
        for id in ids:
            for angle in range(-80,90,10):
                if id+angle*100 in self.stones.keys():
                    del_ids.append(id+angle*100)
        for id in del_ids:
            add_stones.append(self.stones[id])
            del self.stones[id]
        return add_stones


    
    def get_all_stones_replacing(self):
        selected_stones = []
        for key, value in self.stones.copy().items():
            selected_stones.append(value)
        return selected_stones

    def get_variant_stones(self, nb_stones,variant_sample =1):
        if nb_stones < len(self.labels.keys()):
            raise Exception(
                f"Need to command {len(self.labels.keys())-nb_stones} more stones")
        selected_stones = []
        #print("Number of clusters: ", len(self.labels.keys()))
        for label in self.labels.keys():
            if len(self.labels[label]) == 0:
                #print("Cluster ", label, " is empty")
                continue
            #print("Stone id in cluster ", label, ": ", self.labels[label])
            #print("Stones in the data set: ", self.stones.keys())
            self.labels[label] = np.random.permutation(self.labels[label])
            selected_stone_ids_from_label = []
            for i in range(len(self.labels[label])):
                selected_stone_id_from_label = self.labels[label][i]
                if selected_stone_id_from_label in self.stones.keys():
                    selected_stone_ids_from_label.append(selected_stone_id_from_label)
                if len(selected_stone_ids_from_label)==variant_sample:
                    break
            if len(selected_stone_ids_from_label) ==0:
                #print("No stone found for cluster ", label)
                continue
            #print("Selecte id for cluster ", label, ": ", selected_stone_ids_from_label)
            
            for _id in selected_stone_ids_from_label:
                selected_stones.append(
                    self.stones[_id])
                # plt.clf()
                # plt.imshow(self.stones[_id].matrix)
                # plt.show()

                del self.stones[_id]
        return selected_stones

    # def get_similar_stones(self, nb_stones):
    #     # iterate each label, get average area
    #     def get_average_area(cluster_label):
    #         list_of_stone_index = self.labels[cluster_label]
    #         area_sum = 0
    #         nb_stones_rest = 0
    #         for index in list_of_stone_index:
    #             if index in self.stones.keys():
    #                 area_sum += self.stones[index].area
    #                 nb_stones_rest += 1
    #         if nb_stones_rest == 0:
    #             return 0
    #         average_area = area_sum/nb_stones_rest
    #         return average_area

    #     def get_length(cluster_label):
    #         return len(self.labels[cluster_label])
    #     labels = list(self.labels.keys())
    #     #print("Before sorting: ", labels)
    #     labels.sort(key=get_length, reverse=True)
    #     #print("After sorting: ", labels)

    #     selected_stones = []

    #     nb_stones_random = int(np.floor(nb_stones*0.3))
    #     stone_keys = list(self.stones.keys())
    #     if nb_stones_random > len(stone_keys):
    #         get_main_logger().warning("Not enough random stone types from the vendor")
    #         selected_stone_ids = stone_keys
    #     else:
    #         selected_stone_ids = np.random.choice(
    #             stone_keys, size=nb_stones_random, replace=False)
    #     for id in selected_stone_ids:
    #         selected_stones.append(self.stones[id])
    #         del self.stones[id]

    #     for label in labels:
    #         for i in range(len(self.labels[label])):
    #             random_pick = np.random.randint(0, len(self.labels[label]))
    #             selected_stone_id_from_label = self.labels[label][random_pick]
    #             if selected_stone_id_from_label in self.stones.keys() and selected_stone_id_from_label not in selected_stone_ids:
    #                 selected_stones.append(
    #                     self.stones[selected_stone_id_from_label])
    #                 del self.stones[selected_stone_id_from_label]
    #             if len(selected_stones) == nb_stones:
    #                 return selected_stones
    #     return selected_stones
    def get_same_stones(self, nb_stones,pre_label=None,compensate=True, all_same_cluster = False):
        # iterate each label, get average area
        def get_average_area(cluster_label):
            list_of_stone_index = self.labels[cluster_label]
            area_sum = 0
            nb_stones_rest = 0
            for index in list_of_stone_index:
                if index in self.stones.keys():
                    area_sum += self.stones[index].area
                    nb_stones_rest += 1
            if nb_stones_rest == 0:
                return 0
            average_area = area_sum/nb_stones_rest
            return average_area

        def get_length(cluster_label):
            return len(self.labels[cluster_label])
        selected_stones = []
        if pre_label==None:
            #use the largest cluster
            labels = list(self.labels.keys())
            #print("Before sorting: ", labels)
            labels.sort(key=get_length, reverse=True)
            #print("After sorting: ", labels)
            for label in labels:
                for i in range(len(self.labels[label])):
                    random_pick = np.random.randint(0, len(self.labels[label]))
                    selected_stone_id_from_label = self.labels[label][random_pick]
                    if selected_stone_id_from_label in self.stones.keys():
                        selected_stones.append(
                            self.stones[selected_stone_id_from_label])
                        del self.stones[selected_stone_id_from_label]
                    if len(selected_stones) == nb_stones:
                        return selected_stones
        else:
            if all_same_cluster:
                for _label_ in self.labels[pre_label]:
                    if _label_ in self.stones.keys():
                        selected_stones.append(
                            self.stones[_label_])
                        del self.stones[_label_]
                return selected_stones
            else:

                for i in range(len(self.labels[pre_label])):
                    random_pick = np.random.randint(0, len(self.labels[pre_label]))
                    selected_stone_id_from_label = self.labels[pre_label][random_pick]
                    if selected_stone_id_from_label in self.stones.keys():
                        selected_stones.append(
                            self.stones[selected_stone_id_from_label])
                        del self.stones[selected_stone_id_from_label]
                    if len(selected_stones) == nb_stones:
                        return selected_stones

        if compensate:
            nb_stones_random = nb_stones-len(selected_stones)
            stone_keys = list(self.stones.keys())
            if nb_stones_random > len(stone_keys):
                get_main_logger().warning("Not enough random stone types from the vendor")
                selected_stone_ids = stone_keys
            else:
                selected_stone_ids = np.random.choice(
                    stone_keys, size=nb_stones_random, replace=False)
            for id in selected_stone_ids:
                selected_stones.append(self.stones[id])
                del self.stones[id]

        return selected_stones

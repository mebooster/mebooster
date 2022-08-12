# Copyright 2017 Google Inc.

import torch
from tqdm import tqdm

from base_sss import SubsetSelectionStrategy
import numpy as np
import math
from mebooster.utils.kcenter import KCenter, pairwise_distances
import config as cfg
from sklearn.metrics import pairwise_distances

class KCenterGreedyApproach(SubsetSelectionStrategy):
    def __init__(self, size, Y_vec, init_cluster, previous_s=None):
        self.init_cluster = init_cluster
        self.previous_s = previous_s  # Y_copy' id
        super(KCenterGreedyApproach, self).__init__(size, Y_vec)
        self.metric = 'euclidean'
        self.min_distances = None
        self.n_obs = self.Y_vec.shape[0]
        # self.alread_selected = []

    def update_distances(self, cluster_centers=None, cluster_features=None, Y_e=None, only_new=True, reset_dist=False):
        """Update min distances given cluster centers.

            Args:
              cluster_centers: indices of cluster centers
              only_new: only calculate distance for newly selected points and update
                min_distances.
              rest_dist: whether to reset min_distances.
        """
        if reset_dist:
            self.min_distances = None
        if only_new:
            cluster_centers = [d for d in cluster_centers] #self.already_selected is Y_e's index
        if cluster_centers is not None:
            # Update min_distances for all examples given new cluster center.
            # x = self.features[cluster_centers]
            x = Y_e[cluster_centers]
            #don't use self.features
            dist = pairwise_distances(Y_e, x, metric=self.metric) #self.features

            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)
        elif cluster_features is not None:
            dist = pairwise_distances(Y_e, cluster_features, metric=self.metric)  # self.features

            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)

    def get_subset(self):
        if self.previous_s is not None:
            Y_e = np.asarray([self.Y_vec[s1] for s1 in self.previous_s.astype(int)])
            self.n_obs = len(self.previous_s)
        else:
            Y_e = self.Y_vec

        print("self.features,", Y_e.shape)
        print('Calculating distances...')
        self.update_distances(cluster_features=self.init_cluster, Y_e=Y_e, only_new=False, reset_dist=True)# it's features

        new_batch = []
        for _ in range(self.size):
            # if self.alread_selected is None:
            #     ind = np.random.choice(np.arange(self.n_obs))# therefor it's not the true index, but the Y_e's index
            # else:
            ind = np.argmax(self.min_distances)
            # assert ind not in self.init_cluster # therefore already_select is a set of index
            self.update_distances(cluster_centers=[ind], Y_e=Y_e, only_new=True, reset_dist=False)
            new_batch.append(ind)
        print('Maximum distance from cluster centers is %0.2f'
              % max(self.min_distances))

        if self.previous_s is not None:
          final_points = [self.previous_s[int(p)] for p in new_batch]
        else:
          final_points = [int(p) for p in new_batch]
        return final_points  # y_copy's id
"""
MIT License

Copyright (c) 2019 Soham Pal, Yash Gupta, Aditya Shukla, Aditya Kanade,
Shirish Shevade, Vinod Ganapathy. Indian Institute of Science.

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import torch
from tqdm import tqdm

from base_sss import SubsetSelectionStrategy
# import tensorflow as tf
import numpy as np
import math
from mebooster.utils.kcenter import KCenter, pairwise_distances
import config as cfg

class KCenterGreedyApproach(SubsetSelectionStrategy):
    def __init__(self, size, Y_vec, init_cluster, previous_s=None):
        self.init_cluster = init_cluster
        self.previous_s = previous_s  # Y_copy' id
        super(KCenterGreedyApproach, self).__init__(size, Y_vec)

    def get_subset(self):
        if self.previous_s is not None:
            Y_e = [self.Y_vec[s1] for s1 in self.previous_s.astype(int)]
        else:
            Y_e = self.Y_vec
        Y = self.init_cluster

        Y_all = np.vstack((Y, Y_e))
        n_pool = len(Y_all)
        print("Y_all.size,", Y_all.shape)

        lb_flag = np.zeros(n_pool, dtype=bool)
        lb_flag[:len(Y)] = True
        idxs_lb = lb_flag.copy()

        dist_mat = np.matmul(Y_all, Y_all.transpose())
        print("dist_mat,", dist_mat.shape)
        sq = np.array(dist_mat.diagonal()).reshape(n_pool, 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        dist_mat = np.sqrt(dist_mat)
        mat = dist_mat[~lb_flag, :][:, lb_flag] #not labled data 58000 * 2000
        print("mat,", mat.shape)

        # points = []
        with tqdm(total=self.size) as bar:
           for i in range(self.size):
               mat_min = mat.min(axis=1)
               q_idx_ = mat_min.argmax() #this is the label in unlabelled data(Y, and Y_unlabelled)
               q_idx = np.arange(n_pool)[~lb_flag][q_idx_]
               lb_flag[q_idx] = True
               mat = np.delete(mat, q_idx_, 0)
               mat = np.append(mat, dist_mat[~lb_flag, q_idx][:, None], axis=1)
               bar.update(i)

        points = np.arange(n_pool)[(idxs_lb ^ lb_flag)]

        if self.previous_s is not None:
          final_points = [self.previous_s[int(p-len(Y))] for p in points]
        else:
          final_points = [int(p-len(Y)) for p in points]
        return final_points  # y_copy's id
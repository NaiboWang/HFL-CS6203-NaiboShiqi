#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script defines the sampling method for both iid and non-iid setting.
"""

import numpy as np
from torchvision import datasets, transforms


def iid_sampling(dataset, num_users):
    """
    Sample I.I.D. client data
    :param dataset: dataset generated.
    :param num_users: number of users
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def non_iid_sampling(num_users, targets, num_classes = 10,beta=0.4):
    """
    Sample Non-I.I.D. client data
    :param num_users: number of users
    :param num_classes: number of classes
    :param beta: beta for the dirichlet distribution.
    :return: dict of image index
    """
    min_size = 0
    min_require_size = 10
    K = num_classes
    y_train = np.array(targets)
    N = len(y_train)
    if N / 10000 < 1:
        min_require_size = 1
    dict_users = {}

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, num_users))
            # Balance
            proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        dict_users[j] = idx_batch[j]
    return dict_users

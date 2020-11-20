#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script is used to run federated brain algorithm
 by arguments from commandline in single-machine simulation environments.
Please see readme.md to see the supported arguments.
"""

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import *
from models import *
from utils import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu and torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)
    global_model = get_model(args)
    global_model.to(device)
    # Set model to training mode
    global_model.train()
    print(global_model)

    # copy weights
    # state_dict() returns a dictionary containing a whole state of the module
    global_weights = global_model.state_dict()

    train_accuracy, test_accuracy = [], []
    train_losses, test_losses = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0
    # matrix to save the versions of local clients
    version_matrix = np.zeros((args.num_users, args.num_users))
    models = {}
    for i in range(args.num_users):
        models[i] = copy.deepcopy(global_model)
        models[i].train()

    # Training
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_n_samples, local_losses = [], [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()

        for r in range(args.num_users):
            # Select args.num_users * args.frac clients to train at this round/epoch.
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            print("Users selected:", idxs_users)
            for idx in idxs_users:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[idx], logger=logger)
                w, loss, t_model = local_model.update_weights(
                    model=copy.deepcopy(models[idx]), global_round=epoch)
                models[idx].load_state_dict(w)
                version_matrix[idx, idx] = version_matrix[idx, idx] + 1

            idx_user = np.random.choice(range(args.num_users), 1, replace=False)[0]
            v_old = np.reshape(version_matrix[idx_user, :], -1)
            v_new = np.zeros(args.num_users)
            for i in range(args.num_users):
                v_new[i] = version_matrix[i, i]
            # Averaging models
            w_avg = copy.deepcopy(models[idx_user].state_dict())
            # Record how many clients participated in this round of averaging.
            n_participants = 1
            for i in range(args.num_users):
                if v_new[i] > v_old[i]:
                    version_matrix[idx_user, i] = v_new[i]
                    n_participants = n_participants + 1
                    w_model_to_merge = copy.deepcopy(models[i].state_dict())
                    n = len(w_avg.keys())
                    nt = 0
                    for key in w_avg.keys():
                        nt += 1
                        w_avg[key] = additive(w_avg[key], w_model_to_merge[key], args)
            for key in w_avg.keys():
                w_avg[key] = torch.true_divide(w_avg[key], n_participants)
            print("Select user:", idx_user, ", total number of participants:", n_participants, ", process:", r + 1, "/",
                  args.num_users)
            global_model.load_state_dict(w_avg)
            # Update local model versions for selected clients
            version_matrix[idx_user, idx_user] = version_matrix[idx_user, idx_user] + 1
            models[idx_user].load_state_dict(w_avg)
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx_user], logger=logger)
            w, loss, t_model = local_model.update_weights(
                model=copy.deepcopy(models[idx_user]), global_round=epoch)

        # print global training loss every 'print_every' round
        if (epoch + 1) % print_every == 0:
            local_weights = []
            for i in range(args.num_users):
                v_new[i] = version_matrix[i, i]
            print("versions:", v_new)
            for i in range(args.num_users):
                local_weights.append(models[i].state_dict())
                local_n_samples.append(len(user_groups[i]))
            # update global weights
            global_weights = average_weights(local_weights, local_n_samples, args)
            global_model.load_state_dict(global_weights)
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            train_acc, train_loss, _, _ = test_inference(args, global_model, train_dataset)
            print("Accuracy for training set: {}, loss:{} after {} epochs\n".format(train_acc, train_loss, epoch + 1))
            test_acc, test_loss, _, _ = test_inference(args, global_model, test_dataset)
            print("Accuracy for test set: {}, loss:{} after {} epochs\n".format(test_acc, test_loss, epoch + 1))
            train_accuracy.append(train_acc)
            test_accuracy.append(test_acc)
            train_losses.append(train_loss)
            test_losses.append(test_loss)

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/FedBrain_{}_{}_O[{}]_C[{}]_iid[{}]_E[{}]_B[{}]_S[{}].pkl'. \
        format(args.dataset, args.epochs, args.optimizer, args.frac, args.iid, args.local_ep, args.local_bs,
               print_every)
    print("file_name:", file_name)
    with open(file_name, 'wb') as f:
        pickle.dump(
            {"train_accuracy": train_accuracy, "test_accuracy": test_accuracy, "train_losses": train_losses,
             "test_losses": test_losses, "runtime": time.time() - start_time}, f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

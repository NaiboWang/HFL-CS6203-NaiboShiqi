#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script is used to run federated averaging algorithm
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
from utils import *

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

    # Training
    train_accuracy, test_accuracy, client_accuracy = [], [], []
    train_losses, test_losses = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 5
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_n_samples, local_losses = [], [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()
        # Select args.num_users * args.frac clients to train at this round/epoch.
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print("Users selected:", idxs_users)
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss, t_model = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_n_samples.append(len(user_groups[idx]))

        # update global weights by federated averaging algorithm
        global_weights = average_weights(local_weights, local_n_samples, args)

        # load global weights
        global_model.load_state_dict(global_weights)

        # print global training loss every 'print_every' round
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            train_acc, train_loss, _, _ = test_inference(args, global_model, train_dataset)
            print("Accuracy for training set: {}, loss:{} after {} epochs\n".format(train_acc, train_loss, epoch + 1))
            test_acc, test_loss, _, _ = test_inference(args, global_model, test_dataset)
            print("Accuracy for test set: {}, loss:{} after {} epochs\n".format(test_acc, test_loss, epoch + 1))
            temp_weights = global_model.state_dict()
            client_totals = 0
            client_corrects = 0
            for i in range(len(local_weights)):
                global_model.load_state_dict(local_weights[i])
                client_acc, client_loss, client_correct, client_total = test_inference(args, global_model, test_dataset)
                client_totals += client_total
                client_corrects += client_correct
            client_accs = client_corrects / client_totals
            client_accuracy.append(client_accs)
            print("Single client accuracy:", client_accs)
            global_model.load_state_dict(temp_weights)
            train_accuracy.append(train_acc)
            test_accuracy.append(test_acc)
            train_losses.append(train_loss)
            test_losses.append(test_loss)

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/FedAvg_new_{}_{}_O[{}]_C[{}]_iid[{}]_E[{}]_B[{}]_S[{}].pkl'. \
        format(args.dataset, args.epochs, args.optimizer, args.frac, args.iid,
               args.local_ep, args.local_bs, print_every)
    print("file_name:", file_name)
    with open(file_name, 'wb') as f:
        pickle.dump(
            {"train_accuracy": train_accuracy, "test_accuracy": test_accuracy, "train_losses": train_losses,
             "test_losses": test_losses, "runtime": time.time() - start_time, "client_accuracy": client_accuracy}, f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

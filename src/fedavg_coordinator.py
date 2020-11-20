#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script is used to run the coordinator of federated averaging
 by arguments from commandline in real distributed environment.
Please see readme.md to see the supported arguments.
"""
import random
import shlex
import subprocess
import xmlrpc
from xmlrpc.client import ServerProxy
import numpy as np
import os
import copy
import time
import pickle
from tqdm import tqdm
from xmlrpc.server import SimpleXMLRPCServer
import torch
from tensorboardX import SummaryWriter
import json
import shutil

from options import args_parser
from update import *
from utils import *
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    # Load test dataset
    test_dataset = pickle.load(open("../temps/dataset/test_set.pkl", 'rb'))

    if args.gpu and torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))

    device = 'cuda' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'

    if args.gpu >= 0 and torch.cuda.is_available():
        map = 'cuda:'+str(args.gpu)
    else:
        map = 'cpu'

    global_model = get_model(args)
    global_model.to(device)

    # Set model to training mode
    global_model.train()
    print(global_model)

    # copy weights
    # state_dict() returns a dictionary containing a whole state of the module
    global_weights = global_model.state_dict()

    if not os.path.exists("../temps"):
        os.mkdir("../temps")

    workers = []
    with open("../configuration.json", "r", encoding='utf-8') as f:
        new_dict = json.loads(f.read())
    worker_list = new_dict["worker_list"]
    n_samples = []

    # Connected to all clients from the list in configuration.json.
    for i in range(args.num_users):
        while True:
            try:
                worker = ServerProxy(worker_list[i])
                print("Connecting to " + worker_list[i])
                n_samples.append(worker.say_hello(i))
                workers.append(worker)
                break
            except ConnectionRefusedError:
                print("Worker {} not established, retrying...".format(i))
                time.sleep(1)


    training_accuracy, test_accuracy = [], []
    training_losses, test_losses = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 5
    val_loss_pre, counter = 0, 0

    # Training
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses, local_n_samples = [], [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()

        # Select args.num_users * args.frac clients to train at this round/epoch.
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        print("Parallel training on workers:", idxs_users)
        for idx in idxs_users:
            workers[idx].new_round()

        print("Waiting for all selected clients to finish training.")
        loss = 0
        for idx in idxs_users:
            loss += workers[idx].wait_round()
            local_n_samples.append(n_samples[idx])
        print("All selected clients finished training, total loss: {}.".format(loss))

        # Conduct different training process based on whether to use secret sharing.
        if args.secret_share == 0:
            for idx in idxs_users:
                network_start_time = time.time()
                get_handle = open("../temps/local_model_save_" + str(idx) + ".pkl", 'wb')
                get_handle.write(workers[idx].forward_model().data)
                get_handle.close()
                w = torch.load("../temps/local_model_save_" + str(idx) + ".pkl", map_location=map)
                local_weights.append(copy.deepcopy(w))
                print("Model transferring time for model of worker {}: {}s.".format(idx, round(time.time() - network_start_time,5)))

            # update global weights
            global_weights = average_weights(local_weights, local_n_samples, args)
            global_model.load_state_dict(global_weights)

            # Save new global model
            torch.save(global_model.state_dict(), '../temps/global_model.pkl')

            print("Broadcast global model to all not-selected clients.")
            for i in range(len(workers)):
                if i not in idxs_users:
                    put_handle = open('../temps/global_model.pkl', 'rb')
                    workers[i].load_model(xmlrpc.client.Binary(put_handle.read()))
                    put_handle.close()
        else:
            n_total = sum(local_n_samples)

            print("Parallel averaging on workers:", idxs_users)
            for idx in range(args.num_users):
                if idx in idxs_users:
                    workers[idx].new_average_model(1, n_total)
                else:
                    workers[idx].new_average_model(0, 1)

            print("Waiting for all selected clients to finish averaging.")
            for idx in range(args.num_users):
                if idx in idxs_users:
                    workers[idx].wait_average_model(1)
                else:
                    workers[idx].wait_average_model(0)

            network_start_time = time.time()
            get_handle = open("../temps/global_model_save.pkl", 'wb')
            get_handle.write(workers[idxs_users[0]].forward_model().data)
            get_handle.close()
            print("Model transferring time: {}s.".format(round(time.time() - network_start_time,5)))
            print("Global model loaded from worker {}.".format(idxs_users[0]))
            global_weights = torch.load("../temps/global_model_save.pkl", map_location=map)
            global_model.load_state_dict(global_weights)

        # print global training loss every 'print_every' round
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print("Parallel testing on all workers:", idxs_users)
            correct, total, training_loss = 0, 0, 0
            for idx in range(args.num_users):
                workers[idx].new_test()
            print("Waiting for all selected clients to finish testing.")

            for idx in range(args.num_users):
                c, t, l = workers[idx].wait_test()
                correct += c
                total += t
                training_loss += l
                print("Accuracy for the local training set from worker {}: {}".format(idx,round(c/t,4)))
            training_acc = correct / total
            print("Accuracy for the whole training set: {}, loss: {} after {} epochs\n".format(training_acc, training_loss,
                                                                                     epoch + 1))
            test_acc, test_loss = test_inference(args, global_model, test_dataset)
            print("Accuracy for test set: {}, loss:{} after {} epochs\n".format(test_acc, test_loss, epoch + 1))
            training_accuracy.append(training_accuracy)
            training_losses.append(training_loss)
            test_accuracy.append(test_acc)
            test_losses.append(test_loss)

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/FedAvg_coordinator_{}_{}_O[{}]_C[{}]_iid[{}]_E[{}]_B[{}]_S[{}].pkl'. \
        format(args.dataset, args.epochs, args.optimizer, args.frac, args.iid,
               args.local_ep, args.local_bs, print_every)
    print("file_name:", file_name)
    with open(file_name, 'wb') as f:
        pickle.dump(
            {"training_accuracy": training_accuracy, "test_accuracy": test_accuracy, "training_losses": training_losses,
             "test_losses": test_losses, "runtime": time.time() - start_time}, f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

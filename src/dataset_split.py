#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script is used to split the dataset by arguments from commandline for the distributed version.
Accept three arguments:
--dataset
--num_users
--iid
"""
import os
import pickle
import shutil

from options import args_parser
from update import *
from utils import *
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    args = args_parser()

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    if os.path.exists("../temps/dataset"):
        shutil.rmtree('../temps/dataset')
    os.mkdir("../temps/dataset")

    if args.dataset == "COVID19_twitter" or args.dataset == "heartbeat":
        batch_size = 1
    elif args.dataset == "HAM10000":
        batch_size = 32
    else:
        batch_size = args.local_bs

    # partition data by users
    for idx in range(args.num_users):
        print("Generating trainloader for Worker {}.".format(idx))
        train_loader = DataLoader(DatasetSplit(train_dataset, user_groups[idx]), batch_size=batch_size, shuffle=True)
        file_name = "../temps/dataset/train_loader_"+str(idx)+".pkl"
        with open(file_name, 'wb') as f:
            pickle.dump({"train_loader": train_loader}, f)

    print("Generating test set for Coordinator.")
    with open("../temps/dataset/test_set.pkl", 'wb') as f:
        pickle.dump(test_dataset,f)
    print("Dataset Split Done!")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8
"""
This script is used to run the baseline experiments by arguments from commandline.
Please see readme.md to see the supported arguments.
"""
import pickle
import time

from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import *
from options import *
from update import *
from models import *

if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()
    if args.gpu and torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if args.gpu>=0 and torch.cuda.is_available() else 'cpu'

    # load datasets
    train_dataset, test_dataset, _ = get_dataset(args)

    global_model = get_model(args)

    global_model.to(device)

    # Set model to training mode
    global_model.train()
    print("Dataset: ", args.dataset)
    print(global_model)

    # Training
    optimizer = get_optimizer(global_model,args)
    if args.dataset=="COVID19_twitter" or args.dataset == "heartbeat":
        batch_size = 1
    else:
        batch_size = 64
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    train_accuracy, test_accuracy = [], []
    train_losses, test_losses = [], []

    for epoch in tqdm(range(args.epochs)):
        global_model.train()
        for batch_idx, (data, labels) in enumerate(trainloader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = global_model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Train Accuracy
        train_acc, train_loss = test_inference(args, global_model, train_dataset)
        print('\r\nTraining on', len(train_dataset), 'samples')
        print("Training Accuracy: {:.2f}%, loss: {} for epoch {}".format(100 * train_acc, train_loss, epoch + 1))
        # testing
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        print('Test on', len(test_dataset), 'samples')
        print("Test Accuracy: {:.2f}%, loss: {} for epoch {}".format(100 * test_acc, test_loss, epoch + 1))
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/baseline_{}_{}_O[{}]_C[{}]_E[{}]_B[{}].pkl'. \
        format(args.dataset,  args.epochs, args.optimizer, args.frac, args.local_ep, args.local_bs)
    print("file_name:", file_name)
    with open(file_name, 'wb') as f:
        pickle.dump(
            {"train_accuracy": train_accuracy, "test_accuracy": test_accuracy, "train_losses": train_losses, "test_losses": test_losses,  "runtime": time.time() - start_time}, f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

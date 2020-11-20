#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

"""
This script defines the pytorch training process for local clients in simulated environment.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils import get_optimizer


class DatasetSplit(Dataset):
    """
    An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        if args.dataset == "COVID19_twitter" or args.dataset == "heartbeat":
            batch_size = 1
        elif args.dataset == "HAM10000":
            batch_size = 32
        else:
            batch_size = self.args.local_bs
        self.loader = DataLoader(DatasetSplit(dataset, idxs),
                                 batch_size=batch_size, shuffle=True)  # shuffle随机打乱
        self.device = 'cuda' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def update_weights(self, model, global_round=0):
        """
        Update local model weights with local data.
        Args:
            model: model
            global_round: global round.

        Returns:
            model weights, training loss and model it self
        """
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Training
        optimizer = get_optimizer(model, self.args)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (data, labels) in enumerate(self.loader):
                data, labels = data.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(data)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), model


def test_inference(args, model, test_dataset):
    """
    Implementation of test process on local dataset.
    Returns:
        the test accuracy and loss.
    """
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'
    criterion = nn.NLLLoss().to(device)
    if args.dataset == "COVID19_twitter" or args.dataset == "heartbeat":
        batch_size = 1
    elif args.dataset == "HAM10000":
        batch_size = 32
    else:
        batch_size = 128
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for batch_idx, (data, labels) in enumerate(testloader):
        data, labels = data.to(device), labels.to(device)

        # Inference
        outputs = model(data)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    return accuracy, loss

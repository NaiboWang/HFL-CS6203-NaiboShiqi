#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script defines the deep learning models used in all the experiments for all 10 datasets.
"""

from torch import nn
import torch
import torch.nn.functional as F


# MLP Model for Heart Disease Dataset
class MLP_HD(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP_HD, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


# CNN Model for Mnist and FeMnist Dataset
class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# CNN Model for Cifar10 Dataset
class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            # nn.Linear(4096, 1024),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            # nn.Linear(1024, 512),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return F.log_softmax(x, dim=1)


# CNN Model for chest_xray Dataset
class CNNChest(nn.Module):
    def __init__(self, args):
        super(CNNChest, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 128),
            nn.ReLU(inplace=True),
            # nn.Linear(1024, 512),
            # nn.Linear(512, 512),
            # nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return F.log_softmax(x, dim=1)


# CNN Model for COVID19 Dataset
class CNNCOVID19(nn.Module):
    def __init__(self, args):
        # super here used access method of parent class
        # dont worry much just boiler plate
        super(CNNCOVID19, self).__init__()
        # In conv layer in_channels==input; out_channels=output; kernel_size=filter size
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        # Linear layer in_features is input, how 8*31*31 came is explained in above comment
        # out_features=output
        self.fc1 = nn.Linear(in_features=8 * 31 * 31, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=3)

    def forward(self, l):
        # this method implements forward propagation
        # So, layers are structured as such

        # 1 Conv layer may be thinking self.conv1 is an layer object instance how can we call as if it a function
        # Checkout python documents __call__ this special method is used, so that instances behaves like function
        # __call__ this special method invokes anytime the object instance is called. This interacts with forward
        # method.
        l = self.conv1(l)
        l = F.relu(l)
        l = F.max_pool2d(l, kernel_size=2)

        # linear and final layer
        # -1 indicates, give any number of batch size
        l = l.reshape(-1, 8 * 31 * 31)
        l = self.fc1(l)
        l = self.out(l)
        return F.log_softmax(l, dim=1)


# CNN Model for Malaria cell images Dataset
class CNNMalaria(nn.Module):
    def __init__(self, args):
        super(CNNMalaria, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),
            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            # nn.Linear(1024, 128),
            # nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return F.log_softmax(x, dim=1)


# LSTM Model for COVID19_twitter text content Dataset
class CovidTweetSentimentAnalysis(nn.Module):
    def __init__(self, args, drop_prob=0.2):
        super(CovidTweetSentimentAnalysis, self).__init__()
        self.output_size = 2
        self.n_layers = 2
        self.hidden_dim = 256

        self.embedding_layer = nn.Embedding(10663, 400)
        self.lstm = nn.LSTM(400, 256, 2, dropout=drop_prob, batch_first=True)
        if args.gpu >= 0 and torch.cuda.is_available():
            self.hidden = (torch.autograd.Variable(torch.zeros(self.n_layers, 1, self.hidden_dim)).cuda(),
                           torch.autograd.Variable(torch.zeros(self.n_layers, 1, self.hidden_dim)).cuda())
        else:
            self.hidden = (torch.autograd.Variable(torch.zeros(self.n_layers, 1, self.hidden_dim)),
                           torch.autograd.Variable(torch.zeros(self.n_layers, 1, self.hidden_dim)))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        x = x.long()
        embeds = self.embedding_layer(x)
        lstm_out, hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = out[-1].expand(1, self.output_size)

        return F.log_softmax(out, dim=1)

# LSTM Model for heartbeat audio Dataset
class HeartbeatLSTM(nn.Module):
    def __init__(self, args, drop_prob=0.05):
        super(HeartbeatLSTM, self).__init__()
        self.output_size = 3
        self.n_layers = 2
        self.hidden_dim = 64
        self.lstm = nn.LSTM(1, self.hidden_dim, 2, dropout=drop_prob, batch_first=True)
        if args.gpu >= 0 and torch.cuda.is_available():
            self.hidden = (torch.autograd.Variable(torch.zeros(self.n_layers, 1, self.hidden_dim)).cuda(),
                           torch.autograd.Variable(torch.zeros(self.n_layers, 1, self.hidden_dim)).cuda())
        else:
            self.hidden = (torch.autograd.Variable(torch.zeros(self.n_layers, 1, self.hidden_dim)),
                           torch.autograd.Variable(torch.zeros(self.n_layers, 1, self.hidden_dim)))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.hidden_dim, 3)

    def forward(self, x):
        lstm_out, hidden = self.lstm(x, self.hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = out[-1].expand(1, self.output_size)
        out = F.log_softmax(out, dim=1)

        return out

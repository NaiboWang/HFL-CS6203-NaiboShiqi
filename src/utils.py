#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

"""
This script defines major tool functions used in federated learning environments.
"""

import copy
import torchvision

from DatasetGeneration import heartbeat, malaria_cell_images, COVID19, HAM10000, chest_xray, COVID19_twitter, \
    heart_disease
from sampling import *
import secret_share
from models import *

ENV_SET = False
client_sockets = None
count = 0


def get_dataset(args):
    """
    Get dataset based on configurations from arguments.
    Args:
        args: arguments from commandline.
    Returns:
        Train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    num_classes = 10
    targets = None
    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform)
        targets = train_dataset.targets
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=apply_transform)
    elif args.dataset == 'mnist':
        data_dir = '../data/mnist/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)
        targets = train_dataset.targets
        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
    elif args.dataset == "femnist":
        data_dir = '../data/femnist'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                              transform=apply_transform)
        targets = train_dataset.targets
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                             transform=apply_transform)
    elif args.dataset == 'chest_xray':
        train_dataset, test_dataset = chest_xray.generate_dataset()
        num_classes = 2
        targets = train_dataset[:][1]
    elif args.dataset == "malaria_cell_images":
        train_dataset, test_dataset = malaria_cell_images.generate_dataset()
        num_classes = 2
        targets = train_dataset[:][1]
    elif args.dataset == 'COVID19':
        train_dataset, test_dataset = COVID19.generate_dataset()
        num_classes = 3
        targets = train_dataset[:][1]
    elif args.dataset == "COVID19_twitter":
        train_dataset, test_dataset = COVID19_twitter.generate_dataset()
        num_classes = 2
        targets = train_dataset[:][1]
    elif args.dataset == "heart_disease":
        train_dataset, test_dataset = heart_disease.generate_dataset()
        num_classes = 2
        targets = train_dataset[:][1]
    elif args.dataset == "heartbeat":
        train_dataset, test_dataset = heartbeat.generate_dataset()
        num_classes = 3
        targets = train_dataset[:][1]
    elif args.dataset == "HAM10000":
        train_dataset, test_dataset = HAM10000.generate_dataset()
        num_classes = 7
        lesion_type_dict = {'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6}
        targets = [lesion_type_dict[i] if i in lesion_type_dict else i for i in train_dataset.df.values[:, 3]]
    else:
        exit('Error: unrecognized dataset')

    # Split dataset based on iid or non-iid options
    if args.iid == 1:
        user_groups = iid_sampling(train_dataset, args.num_users)
    else:
        if args.dataset == "heart_disease" or args.dataset == "heartbeat":
            print("Dataset too short, use iid sampling.")
            user_groups = iid_sampling(train_dataset, args.num_users)
        else:
            user_groups = non_iid_sampling(args.num_users, targets, num_classes)
    return train_dataset, test_dataset, user_groups


def get_model(args):
    """
    Get models based on arguments from command line/
    Args:
        args: arguments.

    Returns:
        Specific deep learning model.
    """
    # Convolutional neural network
    if args.dataset == 'mnist' or args.dataset == "femnist":
        model = CNNMnist(args=args)
    elif args.dataset == 'cifar':
        model = CNNCifar(args=args)
    elif args.dataset == 'chest_xray':
        model = CNNChest(args=args)
    elif args.dataset == "malaria_cell_images":
        model = CNNMalaria(args=args)
    elif args.dataset == 'COVID19':
        model = CNNCOVID19(args=args)
    elif args.dataset == 'COVID19_twitter':
        model = CovidTweetSentimentAnalysis(args=args)
    elif args.dataset == 'heart_disease':
        model = MLP_HD(dim_in=13, dim_hidden=64, dim_out=2)
    elif args.dataset == "heartbeat":
        model = HeartbeatLSTM(args=args)
    elif args.dataset == "HAM10000":
        model = torchvision.models.densenet121(pretrained=True)
        HAM10000.set_parameter_requires_grad(model, False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 7)
    else:
        exit('Error: unrecognized dataset')

    # Set the model to train and send it to device.
    if args.parallel == 1:
        model = torch.nn.DataParallel(model)
    return model


def get_optimizer(model, args):
    """
    Get optimizer based on dataset and other arguments.
    Args:
        model: deep learning model.
        args: arguments from command line

    Returns:
        Optimizer for PyTorch.
    """
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)
    return optimizer


def average_weights(w, n_samples, args):
    """
    Weights averaging function.
    Args:
        w: weights of all selected clients
        n_samples: number of samples for every client
        args: arguments from commandline

    Returns:
         the average of the weights.
    """
    n_total = sum(n_samples)
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        if args.secret_share == 1:
            print("Secret adding on key {}".format(key))
        w_avg[key] = (n_samples[0] / n_total) * w_avg[key]
        for i in range(1, len(w)):
            w_avg[key] = additive(w_avg[key], (n_samples[i] / n_total) * w[i][key], args)
        # w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def additive(parameterA, parameterB, args):
    """
    Conducting parameter sum.
    Args:
        parameterA: the list of parameter from client A
        parameterB: the list of parameter from client B
        args: arguments from commandline

    Returns:
        Sum results of two parameter lists, i.e., parameterA + parameterB
    """
    global ENV_SET, client_sockets, count
    parameterA_array = parameterA.cpu().numpy()
    parameterB_array = parameterB.cpu().numpy()
    shape = parameterA_array.shape
    parameterA_array_flatten = parameterA_array.flatten()
    parameterB_array_flatten = parameterB_array.flatten()
    #
    if args.secret_share == 1:
        if not ENV_SET:
            client_sockets = secret_share.setup(args.ss_address, [0, 1])
            ENV_SET = True
        data = [parameterA_array_flatten, parameterB_array_flatten]
        sum_result = secret_share.get_secret_sum_multi_process([0, 1], client_sockets, data)
        count += 1
    else:
        sum_result = parameterA_array_flatten + parameterB_array_flatten  # 修改成其他的
    sum_result = np.reshape(sum_result, shape)
    if args.gpu == -1:
        sum_result = torch.from_numpy(sum_result).cpu()
    else:
        sum_result = torch.from_numpy(sum_result).cuda()
    return sum_result


def exp_details(args):
    """
    Explain the details of the experiment
    Args:
        args: arguments from commandline.

    Returns:

    """
    print('\nExperimental details:')
    print(f'    Dataset     : {args.dataset}')
    if args.secret_share == 1:
        s = "True"
    else:
        s = "False"
    print(f'    Using Secret Sharing     : {s}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')

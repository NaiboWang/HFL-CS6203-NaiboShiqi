#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script defines the commandline arguments used in all experiments.
"""
import argparse


def args_parser():
    """
    Define arguments used in experiments.
    Returns:
        Structural arguments list
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--gpu', default=-1, type=int,  help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU (-1).")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--secret_share', type=int, default=0,
                        help="Whether use secret sharing rather than \
                        directly sum")
    parser.add_argument('--worker_number', type=int, default=0,
                        help="worker number")
    parser.add_argument('--address', type=str, default='172.17.0.2',
                        help="address of client")
    parser.add_argument('--ss_address', type=str, default='172.17.0.2',
                        help="address of secret sharing server")

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=100,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=3,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.7,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=5,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=50,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--parallel', type=int, default=0,
                        help='Whether to use multiple GPUs')

    args = parser.parse_args()
    return args

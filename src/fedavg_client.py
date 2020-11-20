#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script is used to run the client of federated averaging
 by arguments from commandline in real distributed environment.
Please see readme.md to see the supported arguments.
"""
import multiprocessing
import struct
import sys
import threading
import xmlrpc
import socket
from socketserver import ThreadingMixIn
from xmlrpc.client import ServerProxy
from xmlrpc.server import SimpleXMLRPCServer
import torch.utils.data as data
from socket import error as SocketError

import numpy as np
import os
import copy
import time
import pickle

import torch

from options import args_parser
from update import *
from utils import *
import warnings

warnings.filterwarnings("ignore")

precision = 2.0 ** 16


# Multi-process Process definition.
class ThreadXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    pass


# Define training thread used for training local models.
class trainThread(threading.Thread):
    def __init__(self, threadID):
        threading.Thread.__init__(self)
        self.threadID = threadID

    def run(self):
        self.result = update_weights()

    def get_result(self):
        try:
            return self.result
        except:
            return None


# Define averaging thread used for averaging local models.
class AvgThread(threading.Thread):
    def __init__(self, threadID, participate=0, n_total=0):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.participate = participate
        self.n_total = n_total

    def run(self):
        update_model(self.participate, self.n_total)


# Define averaging thread used for testing local models.
class testThread(threading.Thread):
    def __init__(self, threadID):
        threading.Thread.__init__(self)
        self.threadID = threadID

    def run(self):
        self.accuracy, self.correct, self.total, self.loss = test_inference()

    def get_result(self):
        try:
            return self.accuracy, self.correct, self.total, self.loss
        except:
            return None


def encode_float_array(arr):
    """
    Encode float array used for socket communication.
    Args:
        arr: float array

    Returns:
        result int list
    """
    arr = arr * precision
    arr = arr.astype(np.int64)
    int_list = arr.tolist()
    return int_list


def decode_float_array(data, size):
    """
    Decode received array.
    Args:
        data: received data.
        size: data size

    Returns:
        Decoded values.
    """
    res = struct.unpack("q" * size, data)
    val = [i / precision for i in res]
    return val


def send(client_socket, data):
    """
    Send data to specific socket client.
    Args:
        client_socket: socket id
        data: data to be sent

    Returns:

    """
    arr_list = encode_float_array(data)
    siz = struct.pack("!I", len(arr_list))
    client_socket.send(siz)
    for val in arr_list:
        dat = struct.pack("q", val)
        client_socket.send(dat)


def receive(client_socket, size):
    """
    Recevie data to specific socket client.
    Args:
        client_socket: socket id
        size: data size to be received

    Returns:
        received data.
    """
    n = 8 * size
    data = bytearray()
    while len(data) < n:
        packet = client_socket.recv(n - len(data))
        if not packet:
            continue
        data.extend(packet)
    assert (n == len(data))
    res = decode_float_array(data, size)
    return res


def get_secret_sum(client_socket, data, return_dict):
    """
    The process to send and receive added sum.
    Args:
        client_socket: socket id
        data: data to be sent.
        return_dict: added sum slot to receive sum.

    Returns:

    """
    size = len(data)
    try:
        send(client_socket, data=data)
        return_dict[0] = receive(client_socket, size=size)
    except SocketError as e:
        print(e)


def close_socket():
    """
    This function is used to close client's socket.
    Returns:

    """
    # send a close signal to cpp server and close its socket
    global client_socket
    signal = struct.pack("!I", 0)
    client_socket.send(signal)
    respond = client_socket.recv(4)
    respond = struct.unpack("!I", respond)
    assert (respond[0] == 0)
    reply = struct.pack("!I", 1)
    client_socket.send(reply)
    client_socket.close()


def update_model(participate, n_total):
    """
    Update local model with all models selected clients.
    Args:
        participate: if to participate this round of averaging.
        n_total: total number of samples used in this averaging round.

    Returns:

    """
    global model, client_socket, worker_number, args, dataset_length
    parameters = model.state_dict()
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    for key in parameters.keys():
        data = parameters[key].cpu().numpy()
        shape = parameters[key].shape
        if participate == 0:
            data = np.zeros(shape=shape)
        data = data.flatten()
        ratio = dataset_length / n_total
        data = data * ratio
        if participate == 1:
            print("Worker {} conducting secret sharing on key {}".format(worker_number, key))
        p = multiprocessing.Process(target=get_secret_sum, args=(client_socket, data, return_dict))
        p.start()
        p.join()
        result = np.reshape(return_dict[0], shape)
        if args.gpu == -1:
            sum_result = torch.from_numpy(result).cpu()
        else:
            sum_result = torch.from_numpy(result).cuda()
        parameters[key] = sum_result
        if participate == 1:
            print("Worker {} finished secret sharing on key {}".format(worker_number, key))

    model.load_state_dict(parameters)


def forward_model():
    """
    Forward the client's local model to the coordinator.
    Returns:
        Binary files of the local model.
    """
    global worker_number
    model_name = '../temps/local_model_' + str(worker_number) + '.pkl'
    torch.save(model.state_dict(), model_name)
    handle = open(model_name, 'rb')
    return xmlrpc.client.Binary(handle.read())


def say_hello(n):
    """
    Say hello when connected by the coordinator.
    Args:
        n: worker number

    Returns:
        The number of samples of local dataset for this client.
    """
    global dataset_length
    print("Hello from worker ", n)
    return dataset_length


def load_model(data):
    """
    Load model from coordinator.
    Args:
        data: received binary model file.

    Returns:

    """
    global model, worker_number
    model_name = "../temps/global_model_by_local_" + str(worker_number) + ".pkl"
    handle = open(model_name, 'wb')
    handle.write(data.data)
    handle.close()
    if args.gpu >= 0 and torch.cuda.is_available():
        map = 'cuda:' + str(args.gpu)
    else:
        map = 'cpu'
    model.load_state_dict(torch.load(model_name, map_location=map))
    print("worker", worker_number, "loaded the global model.")


def new_round():
    """
    Start a new round of training.
    Returns:

    """
    global thread_update, worker_number
    thread_update = trainThread(1)
    thread_update.start()
    print("Worker {} start training.".format(worker_number))


def wait_round():
    """
    Wait for local model to finish training.
    Returns:
        Local loss of current training round.
    """
    global thread_update, worker_number
    thread_update.join()
    loss = thread_update.get_result()
    print("Worker {} training finished, training loss: {}.".format(worker_number, loss))
    return loss


def new_test():
    """
    Start a new test with local model on local dataset.
    Returns:

    """
    global thread_test, worker_number
    thread_test = testThread(3)
    thread_test.start()
    print("Worker {} start testing at local training set.".format(worker_number))


def wait_test():
    """
    Wait for test thread to finish testing.
    Returns:
        the number of samples predicted correct, total number of samples and local test loss.
    """
    global thread_test, worker_number
    thread_test.join()
    accuracy, correct, total, loss = thread_test.get_result()
    print("Worker {} training finished, local training accuracy: {}.".format(worker_number, accuracy))
    return correct, total, loss


def new_average_model(participate=0, n_total=0):
    """
    Start a new averaging thread.
    Returns:

    """
    global worker_number, thread_avg
    thread_avg = AvgThread(2, participate, n_total)
    thread_avg.start()
    if participate == 1:
        print("Worker {} start averaging.".format(worker_number))


def wait_average_model(participate=0):
    """
    Wait for averaging thread to finish averaging.
    Returns:

    """
    global worker_number, thread_avg
    thread_avg.join()
    if participate == 1:
        print("Worker {} average finished.".format(worker_number))


def update_weights():
    """
    Update local model weights with local data.
    Args:
    Returns:
        Local training loss.
    """
    global model, train_loader, criterion, optimizer, args, worker_number
    # Set mode to train model
    model.train()
    epoch_loss = []
    for iter in range(args.local_ep):
        batch_loss = []
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            model.zero_grad()
            log_probs = model(data)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

    return sum(epoch_loss) / len(epoch_loss)


def test_inference():
    """
    Implementation of test process on local dataset
    Returns:
        accuracy of this test round, number of samples predicted correct, total number of samples and local test loss.
    """
    global model, train_loader, device, args, worker_number
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    criterion_test = nn.NLLLoss().to(device)

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        # Inference
        outputs = model(data)
        batch_loss = criterion_test(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total

    return accuracy, correct, total, loss


if __name__ == '__main__':
    BASEPORT = 8080
    args = args_parser()
    thread_update = None
    thread_avg = None
    thread_test = None
    worker_number = int(args.worker_number)
    device = 'cuda' if int(args.gpu) >= 0 and torch.cuda.is_available() else 'cpu'
    model = get_model(args)
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    if not os.path.exists("../temps"):
        os.mkdir("../temps")
    dataset = pickle.load(open("../temps/dataset/train_loader_" + str(worker_number) + ".pkl", 'rb'))
    train_loader = dataset["train_loader"]
    dataset_length = len(train_loader.dataset)
    optimizer = get_optimizer(model, args)

    if args.secret_share == 1:
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((args.ss_address, BASEPORT + worker_number))
        except:
            print("MP-SPDZ server has not started, please start it first by executing the following commands: \r\n cd "
                  "~/HFL_CS6203_NaiboWang_ShiqiZhang/MP-SPDZ;\r\n ./Scripts/cs6203.sh add_by_socket & python run_client.py {}\r\n".format(
                args.num_users))
            print("Then run this experiment again.")
            sys.exit(-1)

    # Start a new XMLRPC Server
    worker = ThreadXMLRPCServer((args.address, 50001 + worker_number), allow_none=True)  # 初始化
    # Register functions
    worker.register_function(say_hello, "say_hello")
    worker.register_function(load_model, "load_model")
    worker.register_function(update_weights, "update_weights")
    worker.register_function(new_round, "new_round")
    worker.register_function(wait_round, "wait_round")
    worker.register_function(wait_average_model, "wait_average_model")
    worker.register_function(new_average_model, "new_average_model")
    worker.register_function(new_test, "new_test")
    worker.register_function(wait_test, "wait_test")
    worker.register_function(forward_model, "forward_model")
    worker.register_function(close_socket, "close_socket")
    print("Worker", worker_number, "started.")
    worker.serve_forever()  # 保持等待调用状态

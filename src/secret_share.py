"""
This script defines the secret share protocol
 to connect with MP-SPDZ library in single-machine simulated environment.
"""
import socket
from socket import error as SocketError
import struct
import numpy as np
import multiprocessing
from multiprocessing import Manager
import sys
import random
import subprocess
import shlex

BASEPORT = 8080
PARTY_SIZE = 3
precision = 2.0 ** 16
CLIENT_PROG_NAME = "secret-add"


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


def setup(address, client_ids):
    """
    Setup socket environment
    Args:
        address: address of MP-SPDZ secret sharing server.
        client_ids: client ids.

    Returns:
        All connect socket ids.
    """
    sockets = []
    for client_id in client_ids:
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((address, BASEPORT + client_id))
            sockets.append(client_socket)
        except:
            print("MP-SPDZ server has not started, please start it first by executing the following commands: \r\n cd "
                  "~/HFL_CS6203_NaiboWang_ShiqiZhang/MP-SPDZ;\r\n ./Scripts/cs6203.sh add_by_socket & python run_client.py 2\r\n")
            print("Then run this experiment again.")
            sys.exit(-1)
    return sockets


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


def get_secret_sum(id, client_socket, data, returned):
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
        returned[id] = receive(client_socket, size=size)
    except SocketError as e:
        print(e)


def get_secret_sum_multi_process(ids, client_sockets, data):
    """
    The process to send and receive added sum with multiprocess.
    Args:
        client_socket: socket id
        data: data to be sent.
        return_dict: added sum slot to receive sum.

    Returns:
        received sum results.
    """
    manager = Manager()
    return_dict = manager.dict()
    jobs = []
    for id in ids:  # for each client, send via socket by multiprocessing
        p = multiprocessing.Process(target=get_secret_sum, args=(id, client_sockets[id], data[id], return_dict))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
    return return_dict[0]


def close(client_sockets):
    """
    This function is used to close all client's socket.
    Returns:

    """
    # send a close signal to cpp server and close its socket
    signal = struct.pack("!I", 0)
    for client_socket in client_sockets:
        client_socket.send(signal)
        respond = client_socket.recv(4)
        respond = struct.unpack("!I", respond)
        assert (respond[0] == 0)
        reply = struct.pack("!I", 1)
        client_socket.send(reply)
        client_socket.close()


if __name__ == '__main__':
    # set up at the beginning of learning
    client_size = int(sys.argv[1])
    ids = [i for i in range(client_size)]
    client_sockets = setup(ids)

    for i in range(5):  # for i-th round
        size = random.randrange(10, 20)
        data = [np.array([1, 2]), np.array([3, 4])]
        result = get_secret_sum_multi_process(ids, client_sockets, data)
        print(result)

    close(client_sockets)

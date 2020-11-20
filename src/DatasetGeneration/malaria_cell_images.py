"""
This script is used to process and generate dataset from original malaria cell images.
"""
from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from PIL import Image
import os

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset


def generate_dataset():
    """
    This function will be invoked by the dataset generation script to get the data tensors used by PyTorch.
    Returns:
        Training and test data tensors for malaria cell images dataset used by PyTorch.
    """
    print("Preparing Dataset.")
    infected = os.listdir('../data/malaria_cell_images/Parasitized/')
    uninfected = os.listdir('../data/malaria_cell_images/Uninfected/')
    data = []
    labels = []
    for i in infected:
        try:
            image = cv2.imread("../data/malaria_cell_images/Parasitized/" + i)
            image_array = Image.fromarray(image, 'RGB')
            resize_img = image_array.resize((64, 64))
            data.append(np.array(resize_img))
            labels.append(1)
        except AttributeError:
            pass

    for u in uninfected:
        try:
            image = cv2.imread("../data/malaria_cell_images/Uninfected/" + u)
            image_array = Image.fromarray(image, 'RGB')
            resize_img = image_array.resize((64, 64))
            data.append(np.array(resize_img))
            labels.append(0)
        except AttributeError:
            pass
    cells = np.array(data)
    labels = np.array(labels)
    n = np.arange(cells.shape[0])
    np.random.shuffle(n)
    cells = cells[n]
    labels = labels[n]
    cells = cells.astype(np.float32)
    labels = labels.astype(np.int32)
    cells = cells / 255
    x_train, x_test, y_train, y_test = train_test_split(cells, labels, test_size=0.3, random_state=111)
    x_train = x_train.reshape(-1, 3, 64, 64)
    x_test = x_test.reshape(-1, 3, 64, 64)
    x_train_tensor = torch.from_numpy(x_train)
    x_train_tensor = x_train_tensor.type(torch.FloatTensor)
    y_train_tensor = torch.from_numpy(y_train)
    y_train_tensor = y_train_tensor.type(torch.LongTensor)
    x_test_tensor = torch.from_numpy(x_test)
    x_test_tensor = x_test_tensor.type(torch.FloatTensor)
    y_test_tensor = torch.from_numpy(y_test)
    y_test_tensor = y_test_tensor.type(torch.LongTensor)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    print("Finished Dataset Processing.")

    return train_dataset, test_dataset


if __name__ == '__main__':
    generate_dataset()

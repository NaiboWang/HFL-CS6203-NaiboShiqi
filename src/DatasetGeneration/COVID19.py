"""
This script is used to process and generate dataset from original COVID19 images.
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import numpy as np  # linear algebra
from keras.preprocessing.image import ImageDataGenerator
import cv2  # pip install opencv-python
import os
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

warnings.filterwarnings('ignore')
from torchvision.datasets import ImageFolder
from torchvision import transforms

labels = ['COVID-19', 'NORMAL', 'Viral-Pneumonia']
img_size = 64


def get_training_data(data_dir):
    """
    This function is used to load data from original files.
    Args:
        data_dir: the directory of original data files.

    Returns:
        structural data from original images
    """
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        img_set = os.listdir(path)
        n = len(img_set)
        for i in range(n):
            try:
                img = img_set[i]
                img_arr = cv2.imread(os.path.join(path, img))
                resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Reshaping images to preferred size
                data.append([resized_arr, class_num])
                if i % 100 == 0:
                    print("Processing images: {}/{}".format(i + 1, n))
            except Exception as e:
                print(e)
    return np.array(data)


def generate_dataset():
    """
    This function will be invoked by the dataset generation script to get the data tensors used by PyTorch.
    Returns:
        Training and test data tensors for COVID19 dataset used by PyTorch.
    """
    if not os.path.exists("../data/COVID-19/COVID-19.npy"):
        print("Processing Training Data.")
        training_data = get_training_data('../data/COVID-19/train')
        print("Processing Test Data.")
        test_data = get_training_data('../data/COVID-19/test')

        x_train, y_train, x_test, y_test = [], [], [], []

        for feature, label in training_data:
            x_train.append(feature)
            y_train.append(label)

        for feature, label in test_data:
            x_test.append(feature)
            y_test.append(label)

        # Normalize the data
        x_train = np.array(x_train) / 255
        x_test = np.array(x_test) / 255

        # resize data for deep learning
        x_train = x_train.reshape(-1, 3, img_size, img_size)
        y_train = np.array(y_train)
        x_test = x_test.reshape(-1, 3, img_size, img_size)
        y_test = np.array(y_test)

        # With data augmentation to prevent overfitting and handling the imbalance in dataset
        dataset = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}
        np.save("../data/COVID-19/COVID-19.npy", dataset)
    else:
        dataset = np.load("../data/COVID-19/COVID-19.npy", allow_pickle=True).item()
        x_train, y_train, x_test, y_test = dataset["x_train"], dataset["y_train"], dataset["x_test"], dataset["y_test"]

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

    return train_dataset, test_dataset


if __name__ == '__main__':
    train_dataset, test_dataset = generate_dataset()
    print("Done!")

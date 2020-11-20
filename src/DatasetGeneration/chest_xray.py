"""
This script is used to process and generate dataset from original chest_xray images.
"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import cv2  # pip install opencv-python
import os
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

labels = ['PNEUMONIA', 'NORMAL']
img_size = 150


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
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
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
        Training and test data tensors for chest_xray dataset used by PyTorch.
    """
    if not os.path.exists("../data/chest_xray/chest_xray.npy"):
        print("Processing Training Data.")
        training_data = get_training_data('../data/chest_xray/train')
        print("Processing Test Data.")
        test_data = get_training_data('../data/chest_xray/test')

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
        x_train = x_train.reshape(-1, 1, img_size, img_size)
        y_train = np.array(y_train)
        x_test = x_test.reshape(-1, 1, img_size, img_size)
        y_test = np.array(y_test)

        # With data augmentation to prevent overfit and handling the imbalance in dataset
        dataset = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}
        np.save("../data/chest_xray/chest_xray.npy", dataset)
    else:
        dataset = np.load("../data/chest_xray/chest_xray.npy", allow_pickle=True).item()
        x_train, y_train, x_test, y_test = dataset["x_train"], dataset["y_train"], dataset["x_test"], dataset["y_test"]

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.2,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(x_train)
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
    generate_dataset()

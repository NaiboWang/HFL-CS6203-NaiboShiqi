"""
This script is used to process and generate dataset from original heart disease csv file.
"""
import sys
import pandas as pd
import numpy as np
import sklearn
import matplotlib
# convert the data to categorical labels
import torch
from keras.utils.np_utils import to_categorical
from sklearn import model_selection
from torch.utils.data import TensorDataset


def generate_dataset():
    """
    This function will be invoked by the dataset generation script to get the data tensors used by PyTorch.
    Returns:
        Training and test data tensors for heart_disease dataset used by PyTorch.
    """
    # read the csv
    cleveland = pd.read_csv('../data/heart_disease/heart.csv')
    # remove missing data (indicated with a "?")
    data = cleveland[~cleveland.isin(['?'])]
    # drop rows with NaN values from DataFrame
    data = data.dropna(axis=0)
    X = np.array(data.drop(['target'], 1))
    y = np.array(data['target'])
    mean = X.mean(axis=0)
    X -= mean
    std = X.std(axis=0)
    X /= std
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, stratify=y, random_state=42,
                                                                        test_size=0.2)

    # torch.from_numpy creates a tensor data from n-d array
    train_data = TensorDataset(torch.from_numpy(X_train).type(torch.FloatTensor),
                               torch.from_numpy(y_train).type(torch.LongTensor))
    test_data = TensorDataset(torch.from_numpy(X_test).type(torch.FloatTensor),
                              torch.from_numpy(y_test).type(torch.LongTensor))

    return train_data, test_data

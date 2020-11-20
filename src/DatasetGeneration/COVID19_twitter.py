"""
This script is used to process and generate dataset from original COVID19 twitter texts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from string import punctuation
from nltk.corpus import stopwords
import nltk
from collections import Counter

warnings.filterwarnings('ignore')


def punctuation_stopwords_removal(sms):
    """
    This function is used to remove stopwords.
    Args:
        sms: input sms text

    Returns:
        SMS text after filtered.
    """
    remove_punctuation = [ch for ch in sms if ch not in punctuation]
    # convert them back to sentences and split into words
    remove_punctuation = "".join(remove_punctuation).split()
    filtered_sms = [word.lower() for word in remove_punctuation if word.lower() not in stopwords.words('english')]
    return filtered_sms


def pad_features(reviews_int, seq_length):
    """
    This function is used to pad 0 to the end os sentences to make the data length equal.
    Args:
        reviews_int: number of reviews
        seq_length: the length of sequence

    Returns:
        features generated.
    """
    features = np.zeros((len(reviews_int), seq_length), dtype=int)
    for i, row in enumerate(reviews_int):
        if len(row) != 0:
            features[i, -len(row):] = np.array(row)[:seq_length]
    return features


def generate_dataset():
    """
    This function will be invoked by the dataset generation script to get the data tensors used by PyTorch.
    Returns:
        Training and test data tensors for COVID19_twitter dataset used by PyTorch.
    """
    nltk.download('stopwords')
    sentiment_df = pd.read_csv('../data/COVID19_twitter/COVID19_twitter.csv')
    sentiment_df.loc[:, 'text'] = sentiment_df['text'].apply(punctuation_stopwords_removal)
    reviews_split = []
    for i, j in sentiment_df.iterrows():
        reviews_split.append(j['text'])
    words = []
    for review in reviews_split:
        for word in review:
            words.append(word)
    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
    encoded_reviews = []
    for review in reviews_split:
        encoded_reviews.append([vocab_to_int[word] for word in review])
    labels_to_int = []
    for i, j in sentiment_df.iterrows():
        if j['sentiment'] == 'joy':
            labels_to_int.append(1)
        else:
            labels_to_int.append(0)
    non_zero_idx = [ii for ii, review in enumerate(encoded_reviews) if len(encoded_reviews) != 0]
    encoded_reviews = [encoded_reviews[ii] for ii in non_zero_idx]
    encoded_labels = np.array([labels_to_int[ii] for ii in non_zero_idx])
    seq_length = 50
    padded_features = pad_features(encoded_reviews, seq_length)

    split_frac = 0.7
    split_idx = int(len(padded_features) * split_frac)

    training_x, remaining_x = padded_features[:split_idx], padded_features[split_idx:]
    training_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

    test_x = remaining_x
    test_y = remaining_y

    # torch.from_numpy creates a tensor data from n-d array
    train_data = TensorDataset(torch.from_numpy(training_x), torch.from_numpy(training_y).type(torch.LongTensor))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y).type(torch.LongTensor))
    return train_data, test_data


if __name__ == '__main__':
    generate_dataset()

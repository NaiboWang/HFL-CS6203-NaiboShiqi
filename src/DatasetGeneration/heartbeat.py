"""
This script is used to process and generate dataset from original heart beat audio files.
"""

import warnings
import torch
from torch.utils.data import TensorDataset
import os
import pandas as pd
import librosa
import librosa.display
import glob
import matplotlib.pyplot as plt
import keras
import IPython.display as ipd
import wave
from scipy.io import wavfile
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os, fnmatch

warnings.filterwarnings("ignore")


def audio_norm(data):
    """
    Normalize audio data
    Args:
        data: original audio data

    Returns:
        normalized data
    """
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data - min_data) / (max_data - min_data + 0.0001)
    return data - 0.5


def load_file_data_without_change(folder, file_names, duration=3, sr=16000):
    """
    Get audio data without padding highest qualify audio
    Args:
        folder: file folders
        file_names: list of file names
        duration: duration set
        sr: audio sampling rate

    Returns:
        structural audio data
    """
    input_length = sr * duration
    data = []
    for file_name in file_names:
        try:
            sound_file = folder + file_name
            print("load file ", sound_file)
            # use kaiser_fast technique for faster extraction
            X, sr = librosa.load(sound_file, res_type='kaiser_fast')
            dur = librosa.get_duration(y=X, sr=sr)
            # extract normalized mfcc feature from data
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T, axis=0)
        except Exception as e:
            print("Error encountered while parsing file: ", file_name)
        feature = np.array(mfccs).reshape([-1, 1])
        data.append(feature)
    return data



def load_file_data(folder, file_names, duration=12, sr=16000):
    """
    get audio data with a fix padding may also chop off some file
    Args:
        folder: file folders
        file_names: list of file names
        duration: duration set
        sr: audio sampling rate
    Returns:
        structural audio data
    """
    input_length = sr * duration
    data = []
    for file_name in file_names:
        try:
            sound_file = folder + file_name
            print("load file ", sound_file)
            # use kaiser_fast technique for faster extraction
            X, sr = librosa.load(sound_file, sr=sr, duration=duration, res_type='kaiser_fast')
            dur = librosa.get_duration(y=X, sr=sr)
            # pad audio file same duration
            if round(dur) < duration:
                print("fixing audio lenght :", file_name)
                y = librosa.util.fix_length(X, input_length)
            # normalized raw audio
            # y = audio_norm(y)
            # extract normalized mfcc feature from data
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T, axis=0)
        except Exception as e:
            print("Error encountered while parsing file: ", file_name)
        feature = np.array(mfccs).reshape([-1, 1])
        data.append(feature)
    return data


def generate_dataset():
    """
    This function will be invoked by the dataset generation script to get the data tensors used by PyTorch.
    Returns:
        Training and test data tensors for heartbeat audio dataset used by PyTorch.
    """
    if not os.path.exists("../data/heartbeat/heartbeat.npy"):
        # parent folder of sound files
        INPUT_DIR = "../data//heartbeat"
        # 16 KHz
        SAMPLE_RATE = 16000
        # seconds
        MAX_SOUND_CLIP_DURATION = 12
        set_a = pd.read_csv(INPUT_DIR + "/set_a.csv")
        set_a_timing = pd.read_csv(INPUT_DIR + "/set_a_timing.csv")
        set_b = pd.read_csv(INPUT_DIR + "/set_b.csv")
        # merge both set-a and set-b
        frames = [set_a, set_b]
        train_ab = pd.concat(frames)
        # get all unique labels
        nb_classes = train_ab.label.unique()
        # Map label text to integer
        CLASSES = ['artifact', 'murmur', 'normal']
        # {'artifact': 0, 'murmur': 1, 'normal': 3}
        NB_CLASSES = len(CLASSES)
        # Map integer value to text labels
        label_to_int = {k: v for v, k in enumerate(CLASSES)}
        # map integer to label text
        int_to_label = {v: k for k, v in label_to_int.items()}
        A_folder = INPUT_DIR + '/set_a/'
        # set-a
        A_artifact_files = fnmatch.filter(os.listdir(INPUT_DIR + '/set_a'), 'artifact*.wav')
        A_artifact_sounds = load_file_data(folder=A_folder, file_names=A_artifact_files,
                                           duration=MAX_SOUND_CLIP_DURATION)
        A_artifact_labels = [0 for items in A_artifact_files]

        A_normal_files = fnmatch.filter(os.listdir(INPUT_DIR + '/set_a'), 'normal*.wav')
        A_normal_sounds = load_file_data(folder=A_folder, file_names=A_normal_files, duration=MAX_SOUND_CLIP_DURATION)
        A_normal_labels = [2 for items in A_normal_sounds]

        A_extrahls_files = fnmatch.filter(os.listdir(INPUT_DIR + '/set_a'), 'extrahls*.wav')
        A_extrahls_sounds = load_file_data(folder=A_folder, file_names=A_extrahls_files,
                                           duration=MAX_SOUND_CLIP_DURATION)
        A_extrahls_labels = [1 for items in A_extrahls_sounds]

        A_murmur_files = fnmatch.filter(os.listdir(INPUT_DIR + '/set_a'), 'murmur*.wav')
        A_murmur_sounds = load_file_data(folder=A_folder, file_names=A_murmur_files, duration=MAX_SOUND_CLIP_DURATION)
        A_murmur_labels = [1 for items in A_murmur_files]

        # test files
        A_unlabelledtest_files = fnmatch.filter(os.listdir(INPUT_DIR + '/set_a'), 'Aunlabelledtest*.wav')
        A_unlabelledtest_sounds = load_file_data(folder=A_folder, file_names=A_unlabelledtest_files,
                                                 duration=MAX_SOUND_CLIP_DURATION)
        A_unlabelledtest_labels = [-1 for items in A_unlabelledtest_sounds]

        # load dataset-b, keep them separate for testing purpose
        B_folder = INPUT_DIR + '/set_b/'
        # set-b
        B_normal_files = fnmatch.filter(os.listdir(INPUT_DIR + '/set_b'), 'normal*.wav')  # include noisy files
        B_normal_sounds = load_file_data(folder=B_folder, file_names=B_normal_files, duration=MAX_SOUND_CLIP_DURATION)
        B_normal_labels = [2 for items in B_normal_sounds]

        B_murmur_files = fnmatch.filter(os.listdir(INPUT_DIR + '/set_b'), 'murmur*.wav')  # include noisy files
        B_murmur_sounds = load_file_data(folder=B_folder, file_names=B_murmur_files, duration=MAX_SOUND_CLIP_DURATION)
        B_murmur_labels = [1 for items in B_murmur_files]

        B_extrastole_files = fnmatch.filter(os.listdir(INPUT_DIR + '/set_b'), 'extrastole*.wav')
        B_extrastole_sounds = load_file_data(folder=B_folder, file_names=B_extrastole_files,
                                             duration=MAX_SOUND_CLIP_DURATION)
        B_extrastole_labels = [1 for items in B_extrastole_files]

        # test files
        B_unlabelledtest_files = fnmatch.filter(os.listdir(INPUT_DIR + '/set_b'), 'Bunlabelledtest*.wav')
        B_unlabelledtest_sounds = load_file_data(folder=B_folder, file_names=B_unlabelledtest_files,
                                                 duration=MAX_SOUND_CLIP_DURATION)
        B_unlabelledtest_labels = [-1 for items in B_unlabelledtest_sounds]
        print("loaded dataset-b")
        # combine set-a and set-b
        x_data = np.concatenate((A_artifact_sounds, A_normal_sounds, A_extrahls_sounds, A_murmur_sounds,
                                 B_normal_sounds, B_murmur_sounds, B_extrastole_sounds))

        y_data = np.concatenate((A_artifact_labels, A_normal_labels, A_extrahls_labels, A_murmur_labels,
                                 B_normal_labels, B_murmur_labels, B_extrastole_labels))

        seed = 1000
        # split data into Train, Validation and Test
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7, random_state=seed,
                                                            shuffle=True)
        dataset = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}
        np.save("../data/heartbeat/heartbeat.npy", dataset)
    else:
        dataset = np.load("../data/heartbeat/heartbeat.npy", allow_pickle=True).item()
        x_train, y_train, x_test, y_test = dataset["x_train"], dataset["y_train"], dataset["x_test"], dataset["y_test"]

    # torch.from_numpy creates a tensor data from n-d array
    train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train).type(torch.LongTensor))
    test_data = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test).type(torch.LongTensor))

    return train_data, test_data


if __name__ == '__main__':
    generate_dataset()

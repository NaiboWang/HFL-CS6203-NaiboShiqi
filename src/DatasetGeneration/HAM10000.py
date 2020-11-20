"""
This script is used to process and generate dataset from original HAM10000 images.
"""
import os, cv2, itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image

# pytorch libraries
import torch
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# sklearn libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def compute_img_mean_std(image_paths):
    """
    Computing the mean and std of three channel on the whole dataset,
        first we should normalize the image from 0-255 to 0-1
    Args:
        image_paths: original paths of images

    Returns:
        mean and std of three channel on the whole dataset
    """

    img_h, img_w = 224, 224
    imgs = []
    means, stdevs = [], []

    for i in tqdm(range(len(image_paths))):
        img = cv2.imread(image_paths[i])
        img = cv2.resize(img, (img_h, img_w))
        imgs.append(img)

    imgs = np.stack(imgs, axis=3)
    print(imgs.shape)

    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # resize to one row
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    return means, stdevs


def set_parameter_requires_grad(model, feature_extracting):
    """
    Set if parameters require grad
    If feature_extract = False, the model is fine-tuned and all model parameters are updated.
    If feature_extract = True, only the last layer parameters are updated, the others remain fixed.
    Args:
        model:
        feature_extracting: a boolean that defines if we are fine-tuning or feature extracting.
    Returns:

    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# Define a pytorch dataloader for this dataset
class HAM10000(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y


def generate_dataset():
    """
    This function will be invoked by the dataset generation script to get the data tensors used by PyTorch.
    Returns:
        Training and test data tensors for HAM10000 dataset used by PyTorch.
    """
    # to make the results are reproducible
    np.random.seed(10)
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    data_dir = "../data/HAM10000"
    all_image_path = glob(os.path.join(data_dir, '*', '*.jpg'))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'dermatofibroma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }
    # norm_mean, norm_std = compute_img_mean_std(all_image_path)
    # Computed in advance
    norm_mean = [0.763038, 0.54564667, 0.57004464]
    norm_std = [0.14092727, 0.15261286, 0.1699712]
    df_original = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata.csv'))
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
    df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
    df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes
    # this will tell us how many images are associated with each lesion_id
    df_undup = df_original.groupby('lesion_id').count()
    # now we filter out lesion_id's that have only one image associated with it
    df_undup = df_undup[df_undup['image_id'] == 1]
    df_undup.reset_index(inplace=True)

    # here we identify lesion_id's that have duplicate images and those that have only one image.
    def get_duplicates(x):
        unique_list = list(df_undup['lesion_id'])
        if x in unique_list:
            return 'unduplicated'
        else:
            return 'duplicated'

    # create a new colum that is a copy of the lesion_id column
    df_original['duplicates'] = df_original['lesion_id']
    # apply the function to this new column
    df_original['duplicates'] = df_original['duplicates'].apply(get_duplicates)
    # now we filter out images that don't have duplicates
    df_undup = df_original[df_original['duplicates'] == 'unduplicated']
    # now we create a test set using df because we are sure that none of these images have augmented duplicates in the train set
    y = df_undup['cell_type_idx']
    _, df_test = train_test_split(df_undup, test_size=0.3, random_state=101, stratify=y)

    # This set will be df_original excluding all rows that are in the test set
    # This function identifies if an image is part of the train or test set.
    def get_test_rows(x):
        # create a list of all the lesion_id's in the test set
        test_list = list(df_test['image_id'])
        if str(x) in test_list:
            return 'test'
        else:
            return 'train'

    # identify train and test rows
    # create a new colum that is a copy of the image_id column
    df_original['train_or_test'] = df_original['image_id']
    # apply the function to this new column
    df_original['train_or_test'] = df_original['train_or_test'].apply(get_test_rows)
    # filter out train rows
    df_train = df_original[df_original['train_or_test'] == 'train']
    # Copy fewer class to balance the number of 7 classes
    data_aug_rate = [15, 10, 5, 50, 0, 40, 5]
    for i in range(7):
        if data_aug_rate[i]:
            df_train = df_train.append([df_train.loc[df_train['cell_type_idx'] == i, :]] * (data_aug_rate[i] - 1),
                                       ignore_index=True)
    # We can split the test set again in a test set and a true test set:
    df_train = df_train.reset_index()
    df_test = df_test.reset_index()

    train_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip(), transforms.RandomRotation(20),
         transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
         transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])

    # define the transformation of the test images.
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                         transforms.Normalize(norm_mean, norm_std)])

    # Define the training set using the table train_df and using our defined transitions (train_transform)
    training_set = HAM10000(df_train, transform=train_transform)
    # Same for the test set:
    test_set = HAM10000(df_test, transform=test_transform)
    return training_set, test_set

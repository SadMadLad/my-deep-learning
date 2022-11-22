from torch.utils.data import Dataset
import pandas as pd
# import os
import imageio
import numpy as np
from sklearn.model_selection import train_test_split


def splitDataset():
    labeler = {
        'airplane': 0,
        'automobile': 1,
        'bird': 2,
        'cat': 3,
        'deer': 4,
        'dog': 5,
        'frog': 6,
        'horse': 7,
        'ship': 8,
        'truck': 9,
    }

    df = pd.read_csv("./dataset/labels.csv")
    df = df.replace({'label': labeler})

    X, y = df.drop(columns=['label']), df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, shuffle=True, stratify=y)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train['id'], X_test['id'], y_train, y_test


X_train, X_test, Y_train, Y_test = splitDataset()


class CIFAR_Test(Dataset):
    def __init__(self, transform=None):
        self.labels = Y_test
        self.images = X_test
        self.transform = transform

    def __getitem__(self, index):
        img = imageio.imread("./dataset/images/" +
                             str(self.images[index]) + ".png")
        img = np.asarray(img)

        if self.transform is not None:
            img = self.transform(img)

        label = self.labels[index]
        return img, label

    def __len__(self):
        return self.labels.shape[0]


class CIFAR_Train(Dataset):
    def __init__(self, transform=None):
        self.labels = Y_train
        self.images = X_train
        self.transform = transform

    def __getitem__(self, index):
        img = imageio.imread("./dataset/images/" +
                             str(self.images[index]) + ".png")
        img = np.asarray(img)

        if self.transform is not None:
            img = self.transform(img)

        label = self.labels[index]
        return img, label

    def __len__(self):
        return self.labels.shape[0]

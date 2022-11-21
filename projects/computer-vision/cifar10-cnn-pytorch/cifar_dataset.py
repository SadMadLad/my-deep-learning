from torch.utils.data import Dataset
import pandas as pd
# import os
import imageio
import numpy as np


class CIFAR(Dataset):
    def __init__(self, transform=None):
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

        self.labels = df['label']
        self.images = df['id']
        self.transform = transform

    def __getitem__(self, index):
        img = imageio.imread("./dataset/images/" + str(self.images[index]) + ".png")
        img = np.asarray(img)

        if self.transform is not None:
            img = self.transform(img)

        label = self.labels[index]
        return img, label

    def __len__(self):
        return self.labels.shape[0]
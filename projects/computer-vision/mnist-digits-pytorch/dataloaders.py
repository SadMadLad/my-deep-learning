from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
import torchvision.transforms as transforms


def Loaders():
    # Loading Datasets

    train_ds = MNIST(root="./dataset/", train=True,
                     transform=transforms.ToTensor())
    test_ds = MNIST(root="./dataset/", train=False,
                    transform=transforms.ToTensor())
    validation_split = 0.15
    total_images = len(train_ds)
    train_ds, validation_ds = random_split(train_ds, lengths=[
                                           total_images - int(validation_split*total_images), int(validation_split*total_images)])

    print("\nTraining samples: ", len(train_ds))
    print("Validation Samples: ", len(validation_ds))
    print("Test Samples: ", len(test_ds), "\n")

    # Preparing Dataloaders

    training_batch_size = 128
    validation_batch_size = 16
    test_batch_size = 32

    train_ds = DataLoader(
        train_ds, batch_size=training_batch_size, shuffle=True)
    validation_ds = DataLoader(
        validation_ds, batch_size=validation_batch_size, shuffle=True)
    test_ds = DataLoader(test_ds, batch_size=test_batch_size, shuffle=True)

    return train_ds, validation_ds, test_ds

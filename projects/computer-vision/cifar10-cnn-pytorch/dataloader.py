from cifar_dataset import CIFAR
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms

def GetSingleBatch(loader):
    for X, y in loader:
        return X, y

def Loader():
    dataset = CIFAR(transform=transforms.ToTensor())
    dataset_size = len(dataset)

    validation_size = int(0.15*dataset_size)
    test_size = int(0.15*dataset_size)
    train_size = dataset_size - validation_size - test_size

    train_ds, validation_ds, test_ds = random_split(
        dataset, lengths=[train_size, validation_size, test_size])

    batch_size = 128

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_ds, batch_size=1000)
    test_loader = DataLoader(test_ds, batch_size=1000)

    return train_loader, validation_loader, test_loader

# Loader()
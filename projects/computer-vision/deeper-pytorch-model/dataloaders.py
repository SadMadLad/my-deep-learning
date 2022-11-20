from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader


def GetSingleBatch(loader):
    for X, y in loader:
        return X, y


def Loader():
    train_ds = MNIST(root="./dataset/", train=True,
                     download=True, transform=transforms.ToTensor())
    valid_ds = MNIST(root="./dataset/", train=False,
                     transform=transforms.ToTensor())

    print("\nTraining Samples: ", len(train_ds))
    print("Validation Samples: ", len(valid_ds))
    train_batch_size = 256
    train_loader = DataLoader(
        train_ds, batch_size=train_batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=1000)

    return train_loader, valid_loader
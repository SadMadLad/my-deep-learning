from cifar_dataset import CIFAR_Train, CIFAR_Test
from torch.utils.data import DataLoader
from torchvision import transforms


def GetSingleBatch(loader):
    for X, y in loader:
        return X, y


def Loader():
    # Transformations

    transformations_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.25),
        transforms.ColorJitter(brightness=2, contrast=2, saturation=1, hue=0.1),
        transforms.RandomRotation(degrees=30),
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(
            0.2023, 0.1994, 0.2010))  # Channel-wise means and stds
    ])
    transformation_test = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(
                                                  0.2023, 0.1994, 0.2010))])

    # Dataset with Transformations

    dataset_train = CIFAR_Train(transform=transformations_train)
    dataset_test = CIFAR_Test(transform=transformation_test)

    batch_size = 400

    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=1000)

    return train_loader, test_loader
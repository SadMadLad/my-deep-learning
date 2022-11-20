from torchvision.datasets import MNIST

def download_data():
    dataset = MNIST(root="dataset/", download=True)
    return

download_data()
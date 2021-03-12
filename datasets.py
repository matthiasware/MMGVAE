import torch
import torchvision


def get_dl_mnist(batch_size, transform_train, transform_valid, file="/mnt/data/pytorch"):
    dl_train = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            file,
            train=True,
            download=True,
            transform=transform_train),
        batch_size=batch_size, shuffle=True)

    dl_valid = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            file,
            train=False,
            download=True,
            transform=transform_valid),
        batch_size=batch_size, shuffle=True)
    return dl_train, dl_valid

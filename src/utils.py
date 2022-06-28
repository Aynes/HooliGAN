from torchvision.datasets import ImageFolder
from torchvision import transforms 
from torch.utils.data import DataLoader
import torch

def get_dataloader(config):
    image_size = config.image_size
    dataroot = config.dataroot
    batch_size = config.batch_size
    workers = config.workers

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = ImageFolder(
        root=dataroot,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=workers,
    )
    return dataloader


def get_device(config):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and config.ngpu > 0) else "cpu")
    return device
from torchvision.datasets import ImageFolder
from torchvision import transforms 
from torch.utils.data import DataLoader
import torch
from torch import nn


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


def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


def prepare_model(config, device, model):
    ngpu = config.ngpu
    model = model.to(device)

    if (device.type == 'cuda') and (ngpu > 1):
        generator = nn.DataParallel(model, list(range(ngpu)))

    model.apply(weights_init)
    return model

def get_criterion(config):
    if config.criterion == 'BCELoss':
        criterion = nn.BCELoss()
    else:
        raise NotImplementedError(f"Criterion [{config.criterion}] not implemented")
    return criterion

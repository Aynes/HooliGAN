from torchvision.datasets import ImageFolder
from torchvision import transforms 
from torch.utils.data import DataLoader
import torch
from torch import nn, optim

from torch.nn.parallel import DataParallel
from models import Generator, Discriminator


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


def init_default_weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


def init_weights(config, model_name, model):
    if config[model_name].weigths is not None:
        path = config[model_name].weigths
        model.load_state_dict(torch.load(path))
    else:
        model.apply(init_default_weights)
    return model


def init_model(config, device, model_name):
    if model_name == 'generator':
        model = Generator(config)
    elif model_name == 'discriminator':
        model = Discriminator(config)
    else:
        raise NotImplementedError(f"model [{model_name}] not implemented")
    
    model = model.to(device)

    return model


def get_model(config, device, model_name):
    ngpu = config.ngpu
    model = init_model(config, device, model_name)
    model = init_weights(config, model_name, model)

    if (device.type == 'cuda') and (ngpu > 1):
        model = DataParallel(model, list(range(ngpu)))

    return model


def get_criterion(config):
    if config.criterion == 'BCELoss':
        criterion = nn.BCELoss()
    else:
        raise NotImplementedError(f"Criterion [{config.criterion}] not implemented")
    return criterion


def get_optimizer(optimizer_config, model):
    if optimizer_config.name == 'Adam':
        lr = optimizer_config[optimizer_config.name].lr
        beta1 = optimizer_config[optimizer_config.name].beta1
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))
    else:
        raise NotImplementedError(f"Optimizer[{optimizer_config.name}] not implemented")
    return optimizer
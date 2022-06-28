from omegaconf import OmegaConf
from utils import get_dataloader, get_device, prepare_model

from models import Generator, Discriminator


PATH_CONFIG = 'config.yml'


config = OmegaConf.load(PATH_CONFIG)
device = get_device(config)

dataloader = get_dataloader(config)

generator = Generator(config)
generator = prepare_model(config, device, generator)

discriminator = Discriminator(config)
discriminator = prepare_model(config, device, discriminator)
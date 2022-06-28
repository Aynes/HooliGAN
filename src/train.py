from omegaconf import OmegaConf
from utils import get_dataloader, get_device, get_generator



PATH_CONFIG = 'config.yml'


config = OmegaConf.load(PATH_CONFIG)
device = get_device(config)

dataloader = get_dataloader(config)

generator = get_generator(config, device)
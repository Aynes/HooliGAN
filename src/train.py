from omegaconf import OmegaConf
from utils import get_dataloader, get_device


PATH_CONFIG = 'config.yml'


config = OmegaConf.load(PATH_CONFIG)
device = get_device
dataloader = get_dataloader(config)
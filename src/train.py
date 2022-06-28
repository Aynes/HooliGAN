from omegaconf import OmegaConf
from utils import get_dataloader


PATH_CONFIG = 'config.yml'


config = OmegaConf.load(PATH_CONFIG)
dataloader = get_dataloader(config)
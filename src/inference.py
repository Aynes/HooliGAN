from omegaconf import OmegaConf
import torch
from torchvision.utils import save_image
from time import time
from utils import get_model, get_device

PATH_CONFIG = 'config.yml'


config = OmegaConf.load(PATH_CONFIG)
device = get_device(config)
generator = get_model(config, device, 'generator')
generator.eval()

noise = torch.randn(3, config.generator.nz, 1, 1, device=device)
fake = generator(noise)
save_image(fake, f'../gen_images/img_{time()git }.png')
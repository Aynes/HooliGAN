from omegaconf import OmegaConf
from utils import get_dataloader, get_device, get_model, get_criterion, get_optimizer
import torchvision.utils as vutils

import torch
import wandb



PATH_CONFIG = '/usr/src/app/src/config.yml'


config = OmegaConf.load(PATH_CONFIG)
run = wandb.init(project=config.project)

device = get_device(config)

dataloader = get_dataloader(config, run)

generator = get_model(config, device, 'generator')
discriminator = get_model(config, device, 'discriminator')

criterion = get_criterion(config)

optimizer_discriminator = get_optimizer(config.discriminator.optimizer, discriminator)
optimizer_generator = get_optimizer(config.generator.optimizer, generator)

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0
real_label = 1.
fake_label = 0.

fixed_noise = torch.randn(64, config.generator.nz, 1, 1, device=device)

for epoch in range(config.num_epochs):
    for i, data in enumerate(dataloader, 0):

        discriminator.zero_grad()

        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

        output = discriminator(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()


        noise = torch.randn(b_size, config.generator.nz, 1, 1, device=device)
        fake = generator(noise)

        label.fill_(fake_label)

        output = discriminator(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake

        optimizer_discriminator .step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        generator.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = discriminator(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizer_generator.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, config.num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == config.num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

torch.save(discriminator.state_dict(), config.generator.weigths)
torch.save(generator.state_dict(), config.discriminator.weigths)

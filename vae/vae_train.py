import torch
import torch.nn as nn
from vae_model import VAE
from pathlib import Path
import skimage.io
import torch
from tqdm import tqdm
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt

PATH_TO_DATASET = '../data/lnnnl'

def KL_divergence(mu, logsigma):
    """
    часть функции потерь, которая отвечает за "близость" латентных представлений разных людей
    """
    loss = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
    return loss

def log_likelihood(x, reconstruction):
    """
    часть функции потерь, которая отвечает за качество реконструкции
    """
    loss = nn.BCELoss(reduction='sum')
    return loss(reconstruction, x)

def loss_vae(x, mu, logsigma, reconstruction):
    return KL_divergence(mu, logsigma) + log_likelihood(x, reconstruction)

def load_data(root_path):
    dimx=64
    dimy=64
    images = []
    for path in Path(root_path).glob("*.jpg"):
        image = skimage.io.imread(path)
        image = resize(image,[dimx,dimy])
        images.append(image)
    return images

    
def train_loop():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    criterion = loss_vae
    autoencoder = VAE(features=16).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters())

    images = load_data(PATH_TO_DATASET)
    train_photos = images
    val_photos = images[:5]

    train_loader = torch.utils.data.DataLoader(train_photos, batch_size=32)
    val_loader = torch.utils.data.DataLoader(val_photos, batch_size=32) 

    n_epochs = 100
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(n_epochs)):
        autoencoder.train()
        train_losses_per_epoch = []
        for batch in train_loader:
            optimizer.zero_grad()
            reconstruction, mu, logsigma = autoencoder(batch.to(device))
            reconstruction = reconstruction.view(-1, 64, 64, 3)
            loss = criterion(batch.to(device).float(), mu, logsigma, reconstruction)
            loss.backward()
            optimizer.step()
            train_losses_per_epoch.append(loss.item())

        train_losses.append(np.mean(train_losses_per_epoch))

        autoencoder.eval()
        val_losses_per_epoch = []
        with torch.no_grad():
            for batch in val_loader:
                reconstruction, mu, logsigma = autoencoder(batch.to(device))
                reconstruction = reconstruction.view(-1, 64, 64, 3)
                loss = criterion(batch.to(device).float(), mu, logsigma, reconstruction)
                val_losses_per_epoch.append(loss.item())

        val_losses.append(np.mean(val_losses_per_epoch))
    torch.save(autoencoder.state_dict(), 'vae.pth')
    
    autoencoder.load_state_dict(torch.load('vae.pth'))
    autoencoder.eval()
    with torch.no_grad():
        for batch in val_loader:
            reconstruction, mu, logsigma = autoencoder(batch.to(device))
            reconstruction = reconstruction.view(-1, 64, 64, 3)
            result = reconstruction.cpu().detach().numpy()
            ground_truth = batch.numpy()
            break


    plt.figure(figsize=(8, 20))
    for i, (gt, res) in enumerate(zip(ground_truth[:5], result[:5])):
        plt.subplot(5, 2, 2*i+1)
        plt.imshow(gt)
        plt.subplot(5, 2, 2*i+2)
        plt.imshow(res)
    plt.savefig('result.png')

    z = np.array([np.random.normal(0, 1, 16) for i in range(10)])
    output = autoencoder.sample(torch.FloatTensor(z).to(device))
    plt.figure(figsize=(18, 18))
    for i in range(output.shape[0]):
        plt.subplot(output.shape[0] // 2, 2, i + 1)
        generated = output[i].cpu().detach().numpy()
        plt.imshow(generated)

    plt.savefig('random.png')

    gt_0 = torch.FloatTensor([ground_truth[0]]).to(device)
    gt_1 = torch.FloatTensor([ground_truth[1]]).to(device)
    first_latent_vector = autoencoder.get_latent_vector(gt_0)
    second_latent_vector = autoencoder.get_latent_vector(gt_1) 

    plt.figure(figsize=(18, 18))
    plt.imshow(autoencoder.sample(first_latent_vector)[0].cpu().detach().numpy())
    plt.savefig('1.png')


    plt.figure(figsize=(18, 18))
    plt.imshow(autoencoder.sample(second_latent_vector)[0].cpu().detach().numpy())
    plt.savefig('2.png')

    plt.figure(figsize=(18, 10))
    for i, alpha in enumerate(np.linspace(0., 1., 10)):
        plt.subplot(1, 10, i + 1)
        latent = (1 - alpha) * first_latent_vector + alpha * second_latent_vector
        img = autoencoder.sample(latent)[0].cpu().detach().numpy()
        plt.imshow(img)
    plt.savefig('inter.png')






if __name__ == "__main__":
    train_loop()
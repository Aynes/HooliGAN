from vae_model import VAE

model = VAE(features=16)
model.load_state_dict(torch.load('vae.pth'))
model.eval()

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

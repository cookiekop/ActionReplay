from vae import VAE
from data_utils import get_data
from torch import optim
import torch

dataset_used = 'MPII'
device = 'cuda'
log_interval = 500
epochs = 50
batch_size = 32
model = VAE(latent_dim=512).to(device)
model_name = 'vae_mark'+str(model.mark)+'_'+dataset_used+'.pth'
train_data_loader, val_data_loader, batch_size, train_size = get_data(dataset_used, batch_size, get_mean_std=False)

optimizer = optim.Adam([
    {'params': model.encoder.parameters()},
    {'params': model.fc_mu.parameters()},
    {'params': model.fc_logvar.parameters()},
    {'params': model.sampler.parameters()},
    {'params': model.decoder.parameters()}
], lr=1e-3)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                             gamma=0.8)

for epoch in range(epochs):
    for i, batch in enumerate(train_data_loader):
        if dataset_used == 'MNIST':
            img, mask = batch[0].to(device), None
        elif dataset_used == 'MPII':
            img, mask = batch['image'].to(device), batch['mask'].to(device)
        gen = model(img)
        loss, recon_loss, kld_loss = model.loss_function(*gen, M_N=batch_size/train_size, mask=mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i % log_interval == log_interval-1):
            print("RECON Loss: {}, KLD loss:{}".format(float(recon_loss.cpu()), float(kld_loss.cpu())))
    scheduler.step()
torch.save(model.state_dict(), 'models/'+model_name)


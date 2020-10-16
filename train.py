from data_utils import UTDVideo, get_data
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from vae import ActionVAE, VAE
from torch import optim
import torch
import json

dataset_used = 'UTDVideo'
device = 'cuda'
log_interval = 100
epochs = 3
batch_size = 1
train_data_loader, val_data_loader, train_size = get_data(dataset_used, batch_size, get_mean_std=False)

VAEmodel = VAE(latent_dim=512).to(device)
VAEmodel_name = 'vae_mark'+str(VAEmodel.mark)+'_UTD'
VAEmodel.load_state_dict(torch.load('models/'+VAEmodel_name+'.pth'))

recurrency = 'rnn'
model = ActionVAE(VAEmodel, recurrency=recurrency, seq_length=5).to(device)
model_name = 'actionvae_'+recurrency+'_mark'+str(model.mark)+'_'+dataset_used

optimizer = optim.Adam([
    {'params': model.encoder.parameters()},
    {'params': model.fc_mu.parameters()},
    {'params': model.fc_logvar.parameters()},
    {'params': model.sampler.parameters()},
    {'params': model.decoder.parameters()},
    {'params': model.rnn.parameters(), 'lr': 1e-3}
], lr=1e-5)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                             gamma=0.95)
losses = []
for epoch in range(epochs):
    for i, batch in enumerate(train_data_loader):
        clip = batch['clip'].to(device)
        gen = model(clip)
        loss, recon_loss, kld_loss = model.loss_function(*gen, M_N=batch_size/train_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i % log_interval == log_interval-1):
            print("RECON Loss: {}, KLD loss:{}".format(float(recon_loss.cpu()), float(kld_loss.cpu())))
    scheduler.step()
    losses.append(float(loss.cpu()))
torch.save(model.state_dict(), 'models/'+model_name+'.pth')
with open('logs/'+model_name+'.json', 'w') as f:
    json.dump(losses, f)
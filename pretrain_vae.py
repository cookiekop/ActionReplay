from vae import VAE
from data_utils import get_data, class_num
from torch import optim
import torch
import json

dataset_used = 'MNIST'
device = 'cuda'
log_interval = 100
epochs = 5
batch_size = 32

model = VAE(latent_dim=100).to(device)
model_name = 'vae_mark'+str(model.mark)+'_'+dataset_used+'_'+str(class_num)
if class_num > 1:
    last_model_name = 'vae_mark'+str(model.mark)+'_'+dataset_used+'_'+str(class_num-1)
    model.load_state_dict(torch.load('models/' + last_model_name + '.pth'))
train_data_loader, val_data_loader, train_size = get_data(dataset_used, batch_size, get_mean_std=False)

optimizer = optim.Adam([
    {'params': model.encoder.parameters(), 'lr': 1e-5},
    {'params': model.fc_mu.parameters()},
    {'params': model.fc_logvar.parameters()},
    {'params': model.sampler.parameters()},
    {'params': model.decoder.parameters()}
], lr=1e-3)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                             gamma=0.95)
losses = []
for epoch in range(epochs):
    epoch_loss = 0.0
    for i, batch in enumerate(train_data_loader):
        if batch[0] is None:
            continue
        if dataset_used in ['MNIST', 'UTD', 'FashionMNIST', 'CIFAR10']:
            img, mask = batch[0].to(device), None
        elif dataset_used == 'MPII':
            img, mask = batch['image'].to(device), batch['mask'].to(device)

        gen = model(img)
        loss, recon_loss, kld_loss = model.loss_function(*gen, M_N=batch_size/train_size, mask=mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += float(loss.cpu())
        if (i % log_interval == log_interval-1):
            print("RECON Loss: {}, KLD loss:{}".format(float(recon_loss.cpu()), float(kld_loss.cpu())))
    scheduler.step()
    losses.append(epoch_loss / i)
torch.save(model.state_dict(), 'models/'+model_name+'.pth')
with open('logs/'+model_name+'.json', 'w') as f:
    json.dump(losses, f)



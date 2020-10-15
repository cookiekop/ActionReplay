from data_utils import UTDVideo, get_data
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from vae import ActionVAE, VAE
from torch import optim
import torch

dataset_used = 'UTDVideo'
device = 'cuda'
log_interval = 100
epochs = 10
batch_size = 32
train_data_loader, val_data_loader, train_size = get_data(dataset_used, batch_size, get_mean_std=False)

VAEmodel = VAE(latent_dim=512).to(device)
VAEmodel_name = 'vae_mark'+str(VAEmodel.mark)+'_UTD'
VAEmodel.load_state_dict(torch.load('models/'+VAEmodel_name+'.pth'))

model = ActionVAE(VAEmodel).to(device)
model_name = 'actionvae_mark'+str(model.mark)+'_'+dataset_used


from classifier import Classifier
from vae import MNIST_VAE
from data_utils import get_data, session_num
from torch import optim
import torch
import json

dataset_used = 'MNIST'
device = 'cuda'
log_interval = 100
epochs = 20
batch_size = 512

vae_model = MNIST_VAE(latent_dim=128).to(device)
vae_model_name = 'gr_vae_' + dataset_used + '_' + str(session_num)

main_model = Classifier(n_class=10).to(device)
main_model_name = 'gr_classifier_' + dataset_used + '_' + str(session_num)
if session_num > 1:
    last_model_name = 'gr_classifier_' + dataset_used + '_' + str(session_num - 1)
    # main_model.load_state_dict(torch.load('models/' + last_model_name + '.pth'))
    last_vae_model_name = 'gr_vae_' + dataset_used + '_' + str(session_num - 1)
    # vae_model.load_state_dict(torch.load('models/' + last_vae_model_name + '.pth'))

    last_main_model = Classifier(n_class=10).to(device)
    last_main_model.load_state_dict(torch.load('models/' + last_model_name + '.pth'))
    last_main_model.eval()
    last_vae_model = MNIST_VAE(latent_dim=128).to(device)
    last_vae_model.load_state_dict(torch.load('models/' + last_vae_model_name + '.pth'))
    last_vae_model.eval()

train_data_loader, val_data_loader, train_size = get_data(dataset_used, batch_size, get_mean_std=False)

optimizer = optim.Adam([
    {'params': main_model.encoder.parameters(), 'lr': 1e-5},
    {'params': main_model.fc1.parameters()},
    {'params': main_model.fc2.parameters()},

    {'params': vae_model.encoder.parameters(), 'lr': 1e-5},
    {'params': vae_model.fc_mu.parameters()},
    {'params': vae_model.fc_logvar.parameters()},
    {'params': vae_model.sampler.parameters()},
    {'params': vae_model.decoder.parameters()}
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
            img, label = batch[0].to(device), batch[1].to(device)
        # elif dataset_used == 'MPII':
        #     img, mask = batch['image'].to(device), batch['mask'].to(device)
        B = -1
        if session_num > 1:
            B = img.shape[0]
            replays = last_vae_model.sample(sample_num=B)
            replay_labels = torch.argmax(last_main_model(replays), dim=1)
            img = torch.cat((img, replays), dim=0)
            label = torch.cat((label, replay_labels), dim=0)

        pred = main_model(img)
        cl_loss = main_model.gr_loss_function(pred, label, session_num)

        gen = vae_model(img)
        gen_loss = vae_model.gr_loss_function(session_num, *gen, N=train_size)

        loss = cl_loss + gen_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        if (i % log_interval == log_interval - 1):
            print("Classifaction Loss: {}, Gen Loss: {}".format(float(cl_loss.cpu()), gen_loss.item()))
    scheduler.step()
    losses.append(epoch_loss / i)

torch.save(main_model.state_dict(), 'models/' + main_model_name + '.pth')
torch.save(vae_model.state_dict(), 'models/' + vae_model_name + '.pth')
with open('logs/' + main_model_name + '.json', 'w') as f:
    json.dump(losses, f)


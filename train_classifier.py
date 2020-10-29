from classifier import Classifier
from data_utils import get_data, session_num
from torch import optim
import torch
import json

dataset_used = 'MNIST'
device = 'cuda'
log_interval = 100
epochs = 5
batch_size = 32

model = Classifier(n_class=10).to(device)
model_name = 'classifier'+'_'+dataset_used+'_'+str(session_num)
if session_num > 1:
    last_model_name = 'classifier'+'_'+dataset_used+'_'+str(session_num-1)
    model.load_state_dict(torch.load('models/' + last_model_name + '.pth'))
train_data_loader, val_data_loader, train_size = get_data(dataset_used, batch_size, get_mean_std=False)

optimizer = optim.Adam([
    {'params': model.encoder.parameters(), 'lr': 1e-5},
    {'params': model.fc1.parameters()},
    {'params': model.fc2.parameters()}
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

        pred = model(img)
        loss = model.loss_function(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += float(loss.cpu())
        if (i % log_interval == log_interval-1):
            print("Loss: {}".format(float(loss.cpu())))
    scheduler.step()
    losses.append(epoch_loss / i)
torch.save(model.state_dict(), 'models/'+model_name+'.pth')
with open('logs/'+model_name+'.json', 'w') as f:
    json.dump(losses, f)

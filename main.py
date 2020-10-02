from data_utils import GeneralVideoDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from vae import ActionVAE
from torch import optim
import torch


batch_size = 1
ucf11 = GeneralVideoDataset('datasets/UCF11_updated_mpg/walking')
data_size = len(ucf11)
print(data_size)
data_loader = DataLoader(ucf11,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=4)

device = 'cuda'
model = ActionVAE(latent_dim=1024,
                  in_channels=3,
                  sequence_length=5)
optim_params = []
for param in model.parameters():
    if param.requires_grad:
        optim_params.append(param)
optimizer = optim.Adam(optim_params,
                       lr=1e-5,
                       weight_decay=0.02)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                             gamma=0.9)
model.to(device)
log_interval = 500
epochs = 1
for epoch in range(epochs):
    for i, batch in enumerate(data_loader):
        vid = batch['clip'].to(device)
        gen = model(vid)
        loss = model.loss_function(*gen,
                                   M_N=batch_size/data_size)['loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Training Loss: {}".format(float(loss.cpu())))
    scheduler.step()
torch.save(model.state_dict(), 'models/actionVAE_mark0.pth')

# model.load_state_dict(torch.load('models/actionVAE_mark0.pth'))
# model.to(device)
# fig = plt.figure()
# for i, vid in enumerate(data_loader):
#     vid = vid['clip'].to(device)
#     # vid = torch.nn.functional.interpolate(vid, size=[3, 128, 128], mode='area').squeeze(0)
#     gen = model.generate(vid).squeeze(0)
#     frame_num = gen.shape[0]
#     for i in range(frame_num):
#         frame = gen[i].permute(1, 2, 0)
#         im = plt.imshow(frame.cpu().detach().numpy())
#         plt.pause(0.1)
#     break
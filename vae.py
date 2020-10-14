import torch
from torch import nn
from torch.nn import functional as F
from types_ import *
from torchvision.models import resnet152, resnet50

class VAE(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 **kwargs) -> None:
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.fc_hidden1 = 1024
        self.fc_hidden2 = 1024
        self.do_p = 0.2
        self.mark = 3

        self.mean = torch.tensor([0.467, 0.445, 0.407])
        self.std = torch.tensor([0.257, 0.252, 0.253])

        # Encoding
        pretrained_net = resnet50(pretrained=True, progress=False)
        modules = list(pretrained_net.children())[:-1]
        modules.extend([nn.Flatten(start_dim=1),
                        nn.Linear(pretrained_net.fc.in_features, self.fc_hidden1),
                        nn.BatchNorm1d(self.fc_hidden1, momentum=0.01),
                        nn.Dropout(p=self.do_p, inplace=True),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.fc_hidden1, self.fc_hidden2),
                        nn.BatchNorm1d(self.fc_hidden2, momentum=0.01),
                        nn.Dropout(p=self.do_p, inplace=True),
                        nn.ReLU(inplace=True)])
        self.encoder = nn.Sequential(*modules)

        # Latent vectors mu and sigma
        self.fc_mu = nn.Linear(self.fc_hidden2, self.latent_dim)
        self.fc_logvar = nn.Linear(self.fc_hidden2, self.latent_dim)

        # Sampling vector
        modules = [
            nn.Linear(self.latent_dim, self.fc_hidden2),
            nn.BatchNorm1d(self.fc_hidden2),
            nn.ReLU(inplace=True),
            nn.Linear(self.fc_hidden2, 64 * 4 * 4),
            nn.BatchNorm1d(64 * 4 * 4),
            nn.ReLU(inplace=True)
        ]
        self.sampler = nn.Sequential(*modules)

        # Decoder
        modules = [
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(16, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid()
        ]
        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        # C = x.shape[1]
        # for i in range(C):
        #     x[:, i, :, :] -= self.mean[i]
        #     x[:, i, :, :] /= self.std[i]
        x = self.encoder(x)

        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)

        return [mu, log_var]

    def decode(self, z):
        z = self.sampler(z).view(-1, 64, 4, 4)
        z = self.decoder(z)
        z = F.interpolate(z, size=(224, 224), mode='area')
        # C = z.shape[1]
        # for i in range(C):
        #     z[:, i, :, :] *= self.std[i]
        #     z[:, i, :, :] += self.mean[i]
        return z

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), x, mu, log_var]

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def loss_function(self, *args, **kwargs):
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']
        mask = kwargs['mask']
        recon_loss = F.binary_cross_entropy(recons, input, weight=mask, reduction='sum')
        kld_loss = torch.sum(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0)
        loss = recon_loss + kld_loss * kld_weight

        return loss, recon_loss, kld_loss

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs):
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

class ActionVAE(VAE):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 sequence_length: int,
                 hidden_dims: List = None):
        super(ActionVAE, self).__init__(in_channels=in_channels,
                                        latent_dim=latent_dim)

        pretrain_resnet = resnet152(pretrained=True)
        modules = list(pretrain_resnet.children())[:-1]
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.sequence_length = sequence_length
        self.rnn = nn.LSTM(input_size=latent_dim,
                          hidden_size=latent_dim,
                          num_layers=3,
                          dropout=0.3,
                          batch_first=True)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        frame_num = input.shape[1]
        frame_0 = input[:, 0]
        mu, log_var = self.encode(frame_0)
        mus, log_vars = mu.unsqueeze(1), log_var.unsqueeze(1)
        zs = self.reparameterize(mu, log_var).unsqueeze(1)
        gen_frames = None
        hn = None
        cn = None

        for frame_idx in range(1, frame_num):
            batch_frame = input[:, frame_idx]
            mu, log_var = self.encode(batch_frame)
            mus, log_vars = torch.cat((mus, mu.unsqueeze(1)), 1), torch.cat((log_vars, log_var.unsqueeze(1)), 1)
            z = self.reparameterize(mu, log_var).unsqueeze(1)
            if (zs.shape[1] < self.sequence_length):
                zs = torch.cat((zs, z), 1)
            else:
                dec_frames, hn, cn = self.decodeVideo(hn, cn, zs)
                if gen_frames == None:
                    gen_frames = dec_frames
                else:
                    gen_frames = torch.cat((gen_frames, dec_frames), 1)
                zs = z
            if (frame_idx == frame_num - 1):
                dec_frames = self.decodeVideo(hn, cn, zs)[0]
                if gen_frames == None:
                    gen_frames = dec_frames
                else:
                    gen_frames = torch.cat((gen_frames, dec_frames), 1)

        return [gen_frames, input, mus, log_vars]

    def decodeVideo(self, hn: Tensor, cn: Tensor, zs: Tensor) -> Tensor:
        gen_frames = None
        if hn == None or cn == None:
            zs, (hn, cn) = self.rnn(zs)
        else:
            zs, (hn, cn) = self.rnn(zs, (hn, cn))
        for seq_idx in range(zs.shape[1]):
            z = zs[:, seq_idx]
            gen_frame = self.decode(z)
            if gen_frames == None:
                gen_frames = gen_frame.unsqueeze(1)
            else:
                gen_frames = torch.cat((gen_frames, gen_frame.unsqueeze(1)), 1)
        return gen_frames, hn, cn

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mus = args[2]
        log_vars = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        input = F.interpolate(input, size=[3, 96, 96], mode="area")

        recons_loss = F.binary_cross_entropy_with_logits(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_vars - mus ** 2 - log_vars.exp(), dim=2))

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]



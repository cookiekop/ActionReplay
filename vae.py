import torch
from torch import nn
from torch.nn import functional as F
from types_ import *
from torchvision.models import resnet50

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
                        #nn.BatchNorm1d(self.fc_hidden1, momentum=0.01),
                        nn.Dropout(p=self.do_p, inplace=True),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.fc_hidden1, self.fc_hidden2),
                        #nn.BatchNorm1d(self.fc_hidden2, momentum=0.01),
                        nn.Dropout(p=self.do_p, inplace=True),
                        nn.ReLU(inplace=True)])
        self.encoder = nn.Sequential(*modules)

        # Latent vectors mu and sigma
        self.fc_mu = nn.Linear(self.fc_hidden2, self.latent_dim)
        self.fc_logvar = nn.Linear(self.fc_hidden2, self.latent_dim)

        # Sampling vector
        modules = [
            nn.Linear(self.latent_dim, self.fc_hidden2),
            #nn.BatchNorm1d(self.fc_hidden2),
            nn.ReLU(inplace=True),
            nn.Linear(self.fc_hidden2, 64 * 4 * 4),
            #nn.BatchNorm1d(64 * 4 * 4),
            nn.ReLU(inplace=True)
        ]
        self.sampler = nn.Sequential(*modules)

        # Decoder
        modules = [
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=(3, 3), stride=(2, 2)),
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

    def sample(self):
        z = torch.randn(1, self.latent_dim)
        z = z.to(next(self.parameters()).device)
        samples = self.decode(z)
        return samples

class ActionVAE(VAE):
    def __init__(self,
                 VAEmodel,
                 **kwargs):
        super(ActionVAE, self).__init__(latent_dim=VAEmodel.latent_dim)
        self.mark = 0
        self.VAE = VAEmodel
        self.encoder = VAEmodel.encoder
        self.sampler = VAEmodel.sampler
        self.fc_mu = VAEmodel.fc_mu
        self.fc_logvar = VAEmodel.fc_logvar
        self.decoder = VAEmodel.decoder

        self.recurrency = kwargs.get('recurrency', None)
        self.seq_length = kwargs.get('seq_length', 5)
        if self.recurrency == 'rnn':
            self.rnn = nn.RNN(input_size=self.latent_dim,
                              hidden_size=self.latent_dim,
                              num_layers=3,
                              nonlinearity='relu',
                              dropout=self.do_p,
                              batch_first=True)
        elif self.recurrency == 'lstm':
            self.rnn = nn.LSTM(input_size=self.latent_dim,
                               hidden_size=self.latent_dim,
                               num_layers=3,
                               dropout=self.do_p,
                               batch_first=True)

    def forward(self, x):
        B, F, C, W, H = x.shape
        zs = torch.zeros_like(x)
        mus = torch.zeros([B, F, self.latent_dim], dtype=x.dtype, device=x.device)
        log_vars = torch.zeros([B, F, self.latent_dim], dtype=x.dtype, device=x.device)
        if self.recurrency == 'rnn':
            states = None
        elif self.recurrency == 'lstm':
            states = (None, None)
        seq_count = 0
        seq = torch.zeros([B, self.seq_length, self.latent_dim], dtype=x.dtype, device=x.device)

        for i in range(F):
            frame = x[:, i]
            if self.recurrency == None:
                z, _, mu, log_var = super(ActionVAE, self).forward(frame)
                zs[:, i] = z
            else:
                mu, log_var = self.encode(frame)
                latent = self.reparameterize(mu, log_var)
                seq[:, seq_count] = latent
                seq_count += 1
                if seq_count == self.seq_length or i == F-1:
                    gen_seq, states = self.decodeSeq(*[seq.clone(), states])
                    zs[:, i + 1 - self.seq_length:i+1] = gen_seq
                    seq_count = 0

            mus[:, i] = mu
            log_vars[:, i] = log_var

        return [zs, x, mus, log_vars]

    def decodeSeq(self, *args):
        seq = args[0]
        states = args[1]
        gen_frames = None
        if type(states) is tuple and states[0] == None:
            output, states = self.rnn(seq)
        else:
            output, states = self.rnn(seq, states)
        for seq_idx in range(output.shape[1]):
            z = output[:, seq_idx]
            gen_frame = self.decode(z)
            if gen_frames == None:
                gen_frames = gen_frame.unsqueeze(1)
            else:
                gen_frames = torch.cat((gen_frames, gen_frame.unsqueeze(1)), 1)
        return gen_frames, states

    def loss_function(self, *args, **kwargs):
        recons = args[0]
        input = args[1]
        mus = args[2]
        log_vars = args[3]

        kld_weight = kwargs['M_N']
        recon_loss = F.binary_cross_entropy(recons, input, reduction='sum')
        kld_loss = torch.sum(-0.5 * torch.sum(1 + log_vars - mus.pow(2) - log_vars.exp(), dim=2))
        loss = recon_loss + kld_loss * kld_weight

        return loss, recon_loss, kld_loss
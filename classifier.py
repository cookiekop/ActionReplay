from torch import nn
from torch.nn import functional as F
from types_ import *
from torchvision.models import resnet50
import torch

class Classifier(nn.Module):
    def __init__(self,
                 n_class: int,
                 **kwargs) -> None:
        super(Classifier, self).__init__()
        self.n_class = n_class
        self.fc_hidden1 = 512
        self.fc_hidden2 = 512

        # Encoding
        modules = [nn.Flatten(start_dim=1),
                   nn.Linear(784, self.fc_hidden1),
                   nn.BatchNorm1d(self.fc_hidden1, momentum=0.01),
                   nn.ReLU(inplace=True),
                   nn.Linear(self.fc_hidden1, self.fc_hidden2),
                   nn.BatchNorm1d(self.fc_hidden2, momentum=0.01),
                   nn.ReLU(inplace=True)]
        self.encoder = nn.Sequential(*modules)

        # FC layer
        self.fc1 = nn.Linear(self.fc_hidden2, 128)
        self.fc2 = nn.Linear(128, self.n_class)

    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(self.fc1(x))
        out = torch.softmax(self.fc2(x), dim=-1)
        return out

    def loss_function(self, pred, target):
        y = torch.eye(self.n_class, device=pred.device)
        loss = F.binary_cross_entropy(pred, y[target], reduction='mean')
        return loss

    def gr_loss_function(self, pred, target, session_num):
        if session_num < 2:
            return self.loss_function(pred, target)
        B = pred.shape[0]
        y = torch.eye(self.n_class, device=pred.device)
        loss_c = F.binary_cross_entropy(pred[:B//2], y[target[:B//2]], reduction='mean')
        loss_r = F.binary_cross_entropy(pred[B//2:], y[target[B//2:]], reduction='mean')
        loss = loss_c / session_num + (1-1/session_num) * loss_r
        return loss


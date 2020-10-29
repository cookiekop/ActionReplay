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
        self.fc_hidden1 = 1024
        self.fc_hidden2 = 512 #1024
        self.do_p = 0.2

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

        # FC layer
        self.fc1 = nn.Linear(self.fc_hidden2, 128)
        self.fc2 = nn.Linear(128, self.n_class)

    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(self.fc1(x))
        out = F.sigmoid(self.fc2(x))
        return out

    def loss_function(self, pred, target):
        y = torch.eye(self.n_class, device=pred.device)
        loss = F.binary_cross_entropy(pred, y[target], reduction='mean')
        return loss
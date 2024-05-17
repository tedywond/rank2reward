import torch
import torch.nn as nn
from torch import distributions
import resnet


model_dict = {'resnet18':resnet.resnet18, 
              'resnet34':resnet.resnet34, 
              'resnet50':resnet.resnet50, 
              'resnet101':resnet.resnet101}


class MLP(nn.Module):
    ''' MLP as used in Vision Transformer, MLP-Mixer and related networks.
    '''
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or hidden_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()
        self.norm = nn.BatchNorm1d(hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        if in_features != out_features:
            self.skip = nn.Linear(in_features, out_features)
        else:
            self.skip = nn.Identity()

    def forward(self, x_input):
        x = self.fc1(x_input)
        x = self.act(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x + self.skip(x_input)


class Model(nn.Module):
    ''' Reward model; takes in a stack of 3 images and outputs a distribution over rewards.
    '''
    def __init__(self, model_type='resnet34', latent_dim=512):
        super().__init__()
        self.model = model_dict[model_type]()
        self.model.fc = nn.Identity()
        self.proj = nn.Sequential(MLP(512 if model_type not in ['resnet50', 'resnet101'] else 2048, latent_dim), 
                                  MLP(latent_dim, latent_dim//2))
        self.fc_mu = nn.Linear(latent_dim//2, 1)
        self.fc_log_var = nn.Linear(latent_dim//2, 1)

    def forward(self, stacked_imgs):
        feat = self.model(stacked_imgs)
        pred = self.proj(feat)
        mu = self.fc_mu(pred)
        log_var = self.fc_log_var(pred)
        std = torch.exp(0.5 * log_var)
        return distributions.Normal(mu, std)

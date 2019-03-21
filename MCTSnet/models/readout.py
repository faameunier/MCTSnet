import torch
import torch.nn.functional as F
import torch.nn as nn


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Rho(nn.Module):
    def __init__(self, embeddings_size=128, n_actions=8):
        super().__init__()
        self.fc1 = nn.Linear(embeddings_size, n_actions).to(device)

    def forward(self, h_s):
        h = self.fc1(h_s.view(h_s.size(0), -1))
        return F.softmax(h, dim=1)


class RhoLu(nn.Module):
    def __init__(self, embeddings_size=128, n_actions=8):
        super().__init__()
        self.fc1 = nn.Linear(embeddings_size, n_actions).to(device)

    def forward(self, h_s):
        h = self.fc1(h_s.view(h_s.size(0), -1))
        return h

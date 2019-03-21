import torch
import torch.nn.functional as F
import torch.nn as nn


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Epsilon(nn.Module):
    def __init__(self, n_features=4, input_shape=(10, 10), embeddings_size=128, subchannels=64):
        super().__init__()
        self.embeddings_size = embeddings_size
        self.input = nn.Conv2d(n_features, subchannels, kernel_size=(3, 3), stride=1, padding=1).to(device)
        self.res1 = nn.Conv2d(subchannels, subchannels, kernel_size=(3, 3), stride=1, padding=1).to(device)
        self.res2 = nn.Conv2d(subchannels, subchannels, kernel_size=(3, 3), stride=1, padding=1).to(device)
        self.res3 = nn.Conv2d(subchannels, subchannels, kernel_size=(3, 3), stride=1, padding=1).to(device)
        self.final = nn.Conv2d(subchannels, subchannels // 2, kernel_size=(1, 1), stride=1, padding=0).to(device)
        self.out = nn.Linear(subchannels // 2 * input_shape[0] * input_shape[1], embeddings_size).to(device)

    def forward(self, state):
        x = F.relu(self.input(state))
        r = F.relu(self.res1(x))
        r = F.relu(self.res2(r))
        r = F.relu(self.res3(r))
        y = x + r
        y = F.relu(self.final(y))
        return F.relu(self.out(y.view((y.size(0), -1))))


class EpsilonC(nn.Module):
    def __init__(self, n_features=4, input_shape=(10, 10), embeddings_size=128, subchannels=64):
        super().__init__()
        self.embeddings_size = embeddings_size
        self.input = nn.Conv2d(n_features, subchannels, kernel_size=(3, 3), stride=1, padding=1).to(device)
        # self.res1 = nn.Conv2d(subchannels, subchannels, kernel_size=(3, 3), stride=1, padding=1).to(device)
        # self.res2 = nn.Conv2d(subchannels, subchannels, kernel_size=(3, 3), stride=1, padding=1).to(device)
        # self.res3 = nn.Conv2d(subchannels, subchannels, kernel_size=(3, 3), stride=1, padding=1).to(device)
        # self.final = nn.Conv2d(subchannels, subchannels // 2, kernel_size=(1, 1), stride=1, padding=0).to(device)
        self.out = nn.Linear(subchannels * input_shape[0] * input_shape[1], embeddings_size).to(device)

    def forward(self, state):
        x = F.relu(self.input(state))
        return F.relu(self.out(x.view((x.size(0), -1))))

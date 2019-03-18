import torch.nn.functional as F
import torch.nn as nn


class PiLogits(nn.Module):
    def __init__(self, embeddings_size=128, n_actions=8):
        super().__init__()
        self.fc1 = nn.Linear(embeddings_size, n_actions)

    def forward(self, h_s):
        h = self.fc1(h_s.view(h_s.size(0), -1))
        return F.relu(h)


class PiPriorLogits(nn.Module):
    def __init__(self, embeddings_size=128, n_actions=8):
        super().__init__()
        self.input = nn.Conv2d(n_actions + 1, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.res1 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.res2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.final = nn.Conv2d(32, 16, kernel_size=(1, 1), stride=1, padding=0)
        self.out = nn.Linear(16 * embeddings_size, n_actions)

    def forward(self, all_h):
        x = F.relu(self.input(all_h))
        r = F.relu(self.res1(x))
        r = F.relu(self.res2(x))
        y = x + r
        y = F.relu(self.final(y))
        return F.relu(self.out(y.view((y.size(0), -1))))


class Pi(nn.Module):
    def __init__(self, embeddings_size=128, n_actions=8, w0=0.5, w1=0.5):
        super().__init__()
        self.embeddings_size = embeddings_size
        self.n_actions = n_actions
        self.piL = PiLogits(embeddings_size, n_actions)
        self.piPL = PiPriorLogits(embeddings_size, n_actions)
        self.w0 = w0
        self.w1 = w1

    def forward(self, all_h):
        psi = self.piL(all_h[:, 0])
        psi_prior = self.piPL(all_h.reshape(-1, self.n_actions + 1, self.embeddings_size, 1))
        return F.softmax(self.w0 * psi + self.w1 * psi_prior, dim=1)

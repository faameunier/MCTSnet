import torch
import torch.nn.functional as F
import torch.nn as nn


class BetaMLP(nn.Module):
    def __init__(self, embeddings_size=128):
        super().__init__()
        self.fc1 = nn.Linear(embeddings_size * 2 + 2, embeddings_size)

    def forward(self, h_i, h_o, reward, action):
        x = torch.cat((h_i, h_o, reward, action), 1)
        h = self.fc1(x.view(x.size(0), -1))
        return F.relu(h)

import torch.nn.functional as F
import torch.nn as nn


class Epsilon(nn.Module):
    def __init__(self, n_features=4, embeddings_size=128):
        super().__init__()
        self.input = nn.Conv2D(input=n_features, filter=64, kernel_size=(3, 3), stride=1, padding=1)
        self.res1 = nn.Conv2D(input=n_features, filter=64, kernel_size=(3, 3), stride=1, padding=1)
        self.res2 = nn.Conv2D(input=n_features, filter=64, kernel_size=(3, 3), stride=1, padding=1)
        self.res3 = nn.Conv2D(input=n_features, filter=64, kernel_size=(3, 3), stride=1, padding=1)
        self.final = nn.Conv2D(input=n_features, filter=32, kernel_size=(1, 1), stride=1, padding=0)
        self.out = nn.Linear(32 * 32 * 4, embeddings_size)

    def forward(self, state):
        x = F.relu(self.input(state))
        r = F.relu(self.res1(x))
        r = F.relu(self.res2(x))
        r = F.relu(self.res3(x))
        y = x + r
        y = F.relu(self.final(y))
        return F.relu(self.out(y.view(-1)))

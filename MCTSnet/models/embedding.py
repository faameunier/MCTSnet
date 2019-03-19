import torch.nn.functional as F
import torch.nn as nn


class Epsilon(nn.Module):
    def __init__(self, n_features=4, input_shape=(10, 10), embeddings_size=128):
        super().__init__()
        self.embeddings_size = embeddings_size
        self.input = nn.Conv2d(n_features, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.res1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.res2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.res3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.final = nn.Conv2d(64, 32, kernel_size=(1, 1), stride=1, padding=0)
        self.out = nn.Linear(32 * input_shape[0] * input_shape[1], embeddings_size)

    def forward(self, state):
        x = F.relu(self.input(state))
        r = F.relu(self.res1(x))
        r = F.relu(self.res2(x))
        r = F.relu(self.res3(x))
        y = x + r
        y = F.relu(self.final(y))
        return F.relu(self.out(y.view((y.size(0), -1))))

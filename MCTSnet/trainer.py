from . import models
from .memory.tree import MemoryTree
import torch
import random
import numpy as np
import gym
import gym_sokoban
# from .sokoban.solver import MCTS


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MCTSnetSokoban():
    def __init__(self, feature_space=(4, 10, 10), n_embeddings=128, n_actions=4, n_simulations=10):
        self.feature_space = feature_space
        self.n_embeddings = n_embeddings
        self.n_actions = n_actions
        self.n_simulations = n_simulations
        self.backup = models.backup.BetaMLP(self.n_embeddings)
        self.embedding = models.embedding.Epsilon(feature_space[0], feature_space[1:], self.n_embeddings)
        self.policy = models.policy.Pi(self.n_embeddings, self.n_actions)
        self.readout = models.readout.Rho(self.n_embeddings, self.n_actions)
        self.model = models.MCTSnet.MCTSnet(self.backup, self.embedding, self.policy, self.readout, self.n_simulations, self.n_actions)
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001)
        self.training_set = []

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, attr):
        self.__model = attr

    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))

    def solve_game(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        env = gym.make("SokobanEnc-v0")
        solution = []
        for s in env.solution:
            state, _, win, _ = env.step(s)
            solution.append({
                'state': state,
                'action': s
            })
            if win:
                break
        return solution

    def train(self, training_size, epochs=10):
        for e in range(epochs):
            running_loss = 0
            ite = 0
            for seed in range(training_size):
                solution = self.solve_game(seed)
                random.seed(seed)
                np.random.seed(seed)
                env = gym.make("SokobanEnc-v0")
                self.model.env = env
                self.model.tree = MemoryTree(self.n_actions)
                for s in solution:
                    ite += 1
                    inputs = torch.tensor(s['state']).to(device)
                    inputs = inputs.reshape((-1, *self.feature_space))

                    temp = [0.0] * self.n_actions
                    temp[s['action'] % 4] = 1.0
                    prediction = torch.tensor([temp]).to(device)

                    # if self.model.tree.get_root() is None:
                    self.model.reset_tree(inputs)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)

                    loss = self.criterion(outputs, prediction)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()

                    # print(s['action'], torch.argmax(outputs))
                    # self.model.replanning(s['action'])
                    if ite % 25 == 24:
                        print('[%d, %5d] mean loss: %.3f' % (e + 1, ite, running_loss / ite))

from . import models
from .memory.tree import MemoryTree
import torch
import random
import numpy as np
import gym
import gym_sokoban
from .mouse import game
from .mouse import solver
from IPython import display
import PIL
import time
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
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0005)
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
                    # /!\ FORGOT TO ADVANCE ENV !!! /!\
                    ite += 1
                    inputs = torch.tensor(s['state']).to(device)
                    inputs = inputs.reshape((-1, *self.feature_space))

                    prediction = torch.tensor([s['action'] % 4]).to(device)

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
                        print('[%d, %5d] mean loss: %.3f' % (e + 1, ite, running_loss / 25))
                        running_loss = 0.0

                    state, _, win, _ = self.model.env.step(s['action'])
                    if win:
                        break


class MCTSnetMouse():
    def __init__(self, feature_space=(2, 10, 10), n_embeddings=256, n_actions=4, n_simulations=25):
        self.feature_space = feature_space
        self.n_embeddings = n_embeddings
        self.n_actions = n_actions
        self.n_simulations = n_simulations
        self.backup = models.backup.Beta(self.n_embeddings)
        self.embedding = models.embedding.Epsilon(feature_space[0], feature_space[1:], self.n_embeddings, subchannels=128)
        self.policy = models.policy.randomPi(n_actions=self.n_actions)
        self.readout = models.readout.RhoLu(self.n_embeddings, self.n_actions)
        self.model = models.MCTSnet.MCTSnet(self.backup, self.embedding, self.policy, self.readout, self.n_simulations, self.n_actions)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
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

    def solve_game(self, seed, max_steps=200):
        random.seed(seed)
        np.random.seed(seed)
        env = game.EnvironmentExploring(temperature=0.3)
        solve = solver.MCTS(env, max_steps=max_steps, n_simulations=10, rollout=100)
        return solve.solve()

    def train(self, training_size, epochs=5, max_steps=200, offset=0):
        for e in range(epochs):
            running_loss = 0
            ite = 0
            for seed in range(training_size):
                solution = self.solve_game(seed + offset, max_steps=max_steps)
                random.seed(seed + offset)
                np.random.seed(seed + offset)
                env = game.EnvironmentExploring(temperature=0.3)
                state = env.reset()
                # display.clear_output(wait=True)
                # display.display(PIL.Image.fromarray(env.get_frame().astype(np.uint8)))
                self.model.env = env
                self.model.tree = MemoryTree(self.n_actions)
                for s in solution:
                    ite += 1
                    inputs = torch.tensor(state).float().to(device)
                    inputs = inputs.reshape((-1, *self.feature_space))

                    # if self.model.tree.get_root() is None:
                    self.model.reset_tree(inputs)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)

                    # objectif = self.compute_values(self.model.tree)
                    # objectif_arg = np.argmax(objectif)
                    objectif2 = torch.tensor([s['scores']]).float().to(device)
                    # objectif = torch.nn.functional.softmax(objectif, dim=1)
                    # print(outputs)
                    loss = self.criterion(outputs, objectif2)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()

                    # print(s['action'], torch.argmax(outputs))
                    # self.model.replanning(s['action'])
                    if ite % max_steps == max_steps - 1:
                        print('[%d, %5d] mean loss: %.3f' % (e + 1, ite, running_loss / max_steps))
                        running_loss = 0.0

                    state, _, win, _ = self.model.env.step(s['action'])
                    if win:
                        break

    def compute_values(self, tree):
        def recursive_explore(node):
            node.value = float(node.reward)
            count = 1
            for c in node.children:
                if c is not None:
                    t_v, t_c = recursive_explore(c)
                    node.value += t_v
                    count += t_c
            return node.value, count

        res = []
        for c in tree.get_root().children:
            if c is not None:
                res.append(recursive_explore(c))
            else:
                res.append((0., 1))
        # print(res)
        # time.sleep(2)
        return [v / c for v, c in res]

    def play(self, seed, max_steps=200):
        random.seed(seed)
        np.random.seed(seed)
        env = game.EnvironmentExploring(temperature=0.3)
        state = env.reset()
        self.model.env = env
        with torch.no_grad():
            first = True
            inputs = torch.tensor(state).float().to(device)
            inputs = inputs.reshape((-1, *self.feature_space))
            self.model.reset_tree(inputs)

            for s in range(max_steps):
                if not first:
                    inputs = torch.tensor(state).float().to(device)
                    inputs = inputs.reshape((-1, *self.feature_space))

                first = False
                self.model.reset_tree(inputs)
                outputs = self.model(inputs)

                self.model.replanning(torch.argmax(outputs))

                state, _, win, _ = self.model.env.step(torch.argmax(outputs))

                display.clear_output(wait=True)
                display.display(PIL.Image.fromarray(env.get_frame().astype(np.uint8)))
                time.sleep(0.1)

                if win:
                    break

        return env.score

    def play_solution(self, seed, max_steps=200):
        solution = self.solve_game(seed, max_steps)
        random.seed(seed)
        np.random.seed(seed)
        env = game.EnvironmentExploring(temperature=0.3)
        env.reset()
        for step in solution:
            display.clear_output(wait=True)
            display.display(PIL.Image.fromarray(env.get_frame().astype(np.uint8)))
            time.sleep(0.05)
            _, _, win, _ = env.step(step['action'])
            if win:
                break
        return env.score

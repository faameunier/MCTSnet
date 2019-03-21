import torch
import torch.nn as nn
from ..memory.tree import *
from .. import utils
import copy


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MCTSnet(nn.Module):
    def __init__(self, backup, embedding, policy, readout, n_simulations=10, n_actions=8):
        super().__init__()
        self.backup = backup
        self.embedding = embedding
        self.policy = policy
        self.readout = readout
        self.n_actions = n_actions
        self.M = n_simulations
        self.tree = None
        self.env = None

    @property
    def env(self):
        return self.__env

    @env.setter
    def env(self, attr):
        self.__env = attr

    def reset_tree(self, x):
        self.tree = MemoryTree(self.n_actions)
        self.tree.set_root(x, self.embedding(x))

    def replanning(self, action):
        self.tree.cut_tree(action)

    def forward(self, x):
        if x.shape[0] > 1:
            raise ValueError("Only a batch size of one is implemented")

        def run_simulation():
            new_env = copy.deepcopy(self.env)
            node = self.tree.get_root()
            next_node = None
            stop = False
            # exploring / exploitation
            while not stop:
                h = node.h
                children = node.children
                h_children = [h]
                for child in children:
                    if child is not None:
                        h_children.append(child.h)
                    else:
                        h_children.append(torch.zeros(self.embedding.embeddings_size, requires_grad=True).reshape(1, self.embedding.embeddings_size).to(device))
                actions = self.policy(torch.cat(h_children, dim=0).reshape(-1, self.policy.n_actions + 1, self.embedding.embeddings_size).to(device))
                next_action = torch.argmax(actions).float().to(device)
                # print(next_action)
                # next_action = utils.softargmax(actions)
                # print(next_action)
                next_node = node.get_child(next_action)
                if next_node is None:
                    # new node discovered
                    state, reward, win, _ = new_env.step(int(next_action))
                    state = torch.tensor([state], requires_grad=True).to(device)
                    temp_node = node.set_child(next_action, state, self.embedding(x), torch.tensor(reward, requires_grad=True).to(device), win)
                    if temp_node is not None:
                        next_node = temp_node
                        stop = True
                    else:
                        node = next_node
                elif next_node.solved:
                    stop = True
                else:
                    node = next_node
            # backup
            stop = False
            while not stop:
                h_t1 = next_node.h
                node = next_node.parent
                if node is None:
                    stop = True
                else:
                    h_t = node.h
                    r_t = next_node.reward
                    a_t = next_node.action
                    node.h = self.backup(h_t, h_t1, r_t.reshape((1, 1)), a_t.reshape((1, 1)))
                    next_node = node

        # run all simulations
        for m in range(self.M):
            run_simulation()

        # readout
        root = self.tree.get_root()
        return self.readout(root.h)

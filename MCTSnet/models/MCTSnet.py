import torch
import torch.nn as nn
from ..memory.tree import *
from ..memory.maze import Maze
from .. import utils
import copy

from IPython import display
import PIL
import numpy as np
import time


DEBUG = False
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MCTSnet(nn.Module):
    """ The actual MCTSnet """

    def __init__(self, backup, embedding, policy, readout, n_simulations=10, n_actions=8, style="copy"):
        super().__init__()
        self.fun = None
        self.style = style
        if style == "copy":
            def helper(env):
                return copy.deepcopy(env)
            self.env_new_sim = helper
        elif style == "set_state":
            def helper(env):
                env.set_state(self.tree.get_root().state.cpu().numpy()[0])
                return env
            self.env_new_sim = helper
        else:
            raise ValueError("Unknown environment copy style")
        self.backup = backup
        self.embedding = embedding
        self.policy = policy
        self.readout = readout
        self.n_actions = n_actions
        self.M = n_simulations
        self.tree = None
        self.env = None

    def reset_tree(self, x):
        self.tree = Maze(self.n_actions)
        self.tree.set_root(x, self.embedding(x))

    def replanning(self, action):
        """ use with caution, if action was never visited, it will raise an error """
        self.tree.cut_tree(action)

    def forward(self, x):
        if x.shape[0] > 1:
            raise ValueError("Only a batch size of one is implemented")

        def run_simulation(env):
            new_env = self.env_new_sim(env)
            self.tree.simulation_reset()

            node = self.tree.get_root()
            next_node = None
            stop = False

            if DEBUG:
                print("Simulation start")
                display.display(PIL.Image.fromarray(new_env.get_frame().astype(np.uint8)))
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
                # next_action = torch.argmax(actions).float().to(device)
                # make the action
                if self.style != "set_state":
                    state, reward, win, _ = new_env.step(int(next_action))
                if DEBUG:
                    display.display(PIL.Image.fromarray(new_env.get_frame().astype(np.uint8)))
                next_node = node.get_child(next_action)

                if next_node is None:  # new node discovered
                    if self.style == "set_state":
                        new_env.set_state(node.state.detach().cpu().numpy()[0])  # set the game to the parent node state
                        state, reward, win, _ = new_env.step(int(next_action))  # make the action
                    # First look the resulting state
                    state = torch.tensor([state], requires_grad=True).to(device)

                    # Add the node to the tree
                    next_node, is_new = node.set_child(next_action, state, self.embedding(x), torch.tensor(reward, requires_grad=True).to(device), win)
                    if is_new:
                        if DEBUG:
                            print("New leaf")
                            print(next_action)
                            display.display(PIL.Image.fromarray(new_env.get_frame().astype(np.uint8)))
                        stop = True
                    else:
                        node = next_node  # the state was known, keep exploring
                elif next_node.solved:  # Winning Leaf
                    stop = True
                else:  # Still exploring the graph
                    node = next_node

            # backup till the root
            stop = False
            # print(self.tree.path)
            while not stop:
                h_t1 = next_node.h
                node = next_node.get_parent()
                if node is None:
                    stop = True
                else:
                    h_t = node.h
                    r_t = next_node.reward
                    a_t = next_node.action
                    node.h = self.backup(h_t, h_t1, r_t.reshape((1, 1)), a_t.reshape((1, 1)))
                    next_node = node

        # run all simulations
        work_env = copy.deepcopy(self.env)
        for m in range(self.M):
            run_simulation(work_env)

        # readout
        root = self.tree.get_root()
        return self.readout(root.h)

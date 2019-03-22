import torch
import torch.nn as nn
from ..memory.tree import *
from .. import utils
import copy


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MCTSnet(nn.Module):
    def __init__(self, backup, embedding, policy, readout, n_simulations=10, n_actions=8, style="copy"):
        super().__init__()
        self.fun = None
        if style == "copy":
            def helper(self):
                return copy.deepcopy(self.env)
            self.fun = helper
        elif style == "set_state":
            def helper(self):
                self.env.set_state(self.tree.get_root().state.cpu().numpy()[0])
                return self.env
            self.fun = helper
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

    @property
    def env(self):
        return self.__env

    @env.setter
    def env(self, attr):
        self.__env = attr

    def reset_tree(self, x):
        self.tree = AcyclicTree(self.n_actions)
        self.tree.set_root(x, self.embedding(x))

    def replanning(self, action):
        self.tree.cut_tree(action)

    def forward(self, x):
        if x.shape[0] > 1:
            raise ValueError("Only a batch size of one is implemented")

        def run_simulation():
            new_env = self.fun(self)
            node = self.tree.get_root()
            next_node = None
            stop = False
            # exploring / exploitation
            deadend_unlock = 0
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
                if next_node is None:  # new node discovered
                    # First look the resulting state
                    state, reward, win, _ = new_env.step(int(next_action))
                    state = torch.tensor([state], requires_grad=True).to(device)

                    # Try to add the node to the tree
                    temp_node = node.set_child(next_action, state, self.embedding(x), torch.tensor(reward, requires_grad=True).to(device), win)
                    if temp_node is not None:
                        # The node was added to the tree succesfully, therefore it is a leaf
                        next_node = temp_node
                        stop = True
                    else:
                        # The node was rejected, are any moves still possible ?
                        if not node.moves:
                            # No more moves to perform, this is a deadend, stop simulation
                            next_node = node
                            stop = True
                        else:
                            # Yes, this is not a deadend yet, try again
                            deadend_unlock += 1
                            if deadend_unlock >= self.n_actions * 8:
                                # The policy keeps requesting already visited states
                                # Stop the simulation (easy solution for such problem)
                                next_node = node
                                stop = True
                            else:
                                # Try again policy
                                node = next_node
                elif next_node.solved:  # Winning Leaf
                    stop = True
                else:  # Still exploring the graph
                    # Reset the deadend_unlock variable
                    deadend_unlock = 0
                    node = next_node
            # backup till the root
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

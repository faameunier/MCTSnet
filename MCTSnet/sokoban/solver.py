import numpy as np
from . import utils
import copy
import random
from tqdm import tqdm
import marshal


class MCTSTree():
    def __init__(self):
        self.root = None
        # self.nodes = []
        # self.nodes_hash = []

    def set_root(self, state):
        if self.root is None:
            self.root = MCTSNode(state)
            # self.nodes.append(self.root)
            # self.nodes_hash.append(marshal.dumps(self.root))
        else:
            raise ValueError("Root is already defined")

    def get_root(self):
        return self.root

    def cut_tree(self, root_action):
        new_root = self.root.children[root_action]
        for k in reversed(range(4)):
            if k != root_action:
                del self.root.children[k]
        new_root.parent = None
        del self.root
        self.root = new_root


class MCTSNode():
    def __init__(self, state, reward=0, finished=False, action=None, parent=None, tree=None):
        self.action = action
        self.state = state
        self.reward = reward
        self.parent = parent
        self.finished = finished
        self.tree = tree
        if not self.finished:
            self.moves = [0, 1, 2, 3]
        else:
            self.moves = []
        self.children = [None for k in self.moves]
        self.value = reward
        self.visits = 1

    def set_child(self, action, state, reward, finished):
        node = MCTSNode(state, reward, finished, action, parent=self)
        self.moves.remove(action)
        self.children[action] = node
        return node

    def get_best_child(self):
        s = sorted(self.children, key=lambda n: n.value / n.visits + np.sqrt(2 * np.log(self.visits) / n.visits))[-1]
        return s

    def update(self, value):
        self.visits += 1
        self.value += value


class MCTS():
    def __init__(self, env, max_steps=100, n_simulations=10, rollout=100):
        self.env = env
        self.n_simulations = n_simulations
        self.max_steps = max_steps
        self.rollout = rollout
        self.tree = MCTSTree()
        self.tree.set_root(utils.encode(env.room_state))
        self.solution = []

    def run_sim(self):
        env = copy.deepcopy(self.env)
        node = self.tree.get_root()
        state = None
        reward = None
        win = None

        while node.moves == [] and node.children != []:
            node = node.get_best_child()
            state, reward, win, _ = env.step(node.action)

        if not node.finished:
            action = random.choice(node.moves)
            state, reward, win, _ = env.step(action)
            node = node.set_child(action, state, reward, win)

        for k in range(self.rollout):
            state, reward, win, _ = env.step(random.choice([0, 1, 2, 3]))
            if win:
                break

        while node is not None:
            node.update(reward)
            node = node.parent

    def find_next_action(self):
        for s in range(self.n_simulations):
            self.run_sim()
        children = self.tree.get_root().children
        temp = []
        for c in children:
            if c is not None:
                temp.append(c)
        return sorted(temp, key=lambda n: n.visits)[-1].action

    def solve(self):
        self.solution = []
        for k in tqdm(range(self.max_steps)):
            action = self.find_next_action()
            _, _, win, _ = self.env.step(action)
            self.solution.append(
                {
                    'state': self.tree.get_root().state,
                    'action': action
                })
            self.tree.cut_tree(action)
            if win:
                break
        return self.solution

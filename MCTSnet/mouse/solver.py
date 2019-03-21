import numpy as np
import copy
import random
from tqdm import tqdm
import pickle


class MCTSTree():
    def __init__(self):
        self.root = None
        self.visited_states = []
        # self.nodes = []
        # self.nodes_hash = []

    def set_root(self, state):
        if self.root is None:
            self.visited_states.append(pickle.dumps(state))
            self.root = MCTSNode(state, tree=self)
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
        self.reward = reward
        self.parent = parent
        self.finished = finished
        self.state = state
        self.tree = tree
        if not self.finished:
            self.moves = [0, 1, 2, 3]
        else:
            self.moves = []
        self.children = [None for k in self.moves]
        self.value = reward
        self.visits = 1

    def set_child(self, state, action, reward, finished):
        self.moves.remove(action)
        temp = pickle.dumps(state)
        if temp in self.tree.visited_states:
            return None
        self.tree.visited_states.append(temp)
        node = MCTSNode(state, reward, finished, action, parent=self, tree=self.tree)
        self.children[action] = node
        return node

    def get_best_child(self):
        scores = []
        for n in self.children:
            if n is not None:
                scores.append(n.value / n.visits + np.sqrt(2 * np.log(self.visits) / n.visits))
            else:
                scores.append(-np.inf)
        return self.children[np.argmax(scores)]

    def update(self, value):
        self.visits += 1
        self.value += value


class MCTS():
    def __init__(self, env, max_steps=100, n_simulations=25, rollout=50, verbose=True):
        self.env = env
        self.n_simulations = n_simulations
        self.max_steps = max_steps
        self.rollout = rollout
        self.tree = MCTSTree()
        state = env.reset()
        self.tree.set_root(state)
        self.solution = []
        self.verbose = verbose

    def run_sim(self):
        env = copy.deepcopy(self.env)
        node = self.tree.get_root()
        state = None
        reward = None
        win = None

        while node.moves == [] and not node.finished:
            node = node.get_best_child()
            state, reward, win, _ = env.step(node.action)

        if not node.finished:
            while True:
                if not node.moves:
                    node.finished = True
                    break
                temp_env = copy.deepcopy(env)
                action = random.choice(node.moves)
                state, reward, win, _ = temp_env.step(action)
                new_node = node.set_child(state, action, reward, win)
                if new_node is not None:
                    env.step(action)
                    node = new_node
                    del temp_env
                    break

        for k in range(self.rollout):
            state, reward, win, _ = env.step(random.choice([0, 1, 2, 3]))
            if win:
                break

        while node is not None:
            node.update(reward)
            node = node.parent

    def find_next_action(self, scores=False):
        for s in range(self.n_simulations):
            self.run_sim()
        if not scores:
            return self.tree.get_root().get_best_child().action
        else:
            root = self.tree.get_root()
            scores = []
            scores = []
            for n in root.children:
                if n is not None:
                    scores.append(n.value / n.visits + np.sqrt(2 * np.log(root.visits) / n.visits))
                else:
                    scores.append(-10)
            # print(scores)
            return np.argmax(scores), scores

    def solve(self):
        self.solution = []
        for k in tqdm(range(self.max_steps), disable=self.verbose):
            action, scores = self.find_next_action(True)
            state, _, win, _ = self.env.step(action)
            self.solution.append(
                {
                    'action': action,
                    'scores': scores,
                })
            self.tree = MCTSTree()
            self.tree.set_root(state)
            # self.tree.cut_tree(action)
            if win:
                break
        return self.solution

import pickle


class MemoryTree():
    def __init__(self, n_children):
        self.n_children = n_children
        self.root = None

    def set_root(self, state, h):
        if self.root is None:
            self.root = MemoryNode(state, h, 0, False, self)
        else:
            raise ValueError("Root is already defined")

    def get_root(self):
        return self.root

    def cut_tree(self, root_action):
        new_root = self.root.children[root_action]
        for k in reversed(range(self.n_children)):
            if k != root_action:
                del self.root.children[k]
        if new_root is not None:
            new_root.parent = None
        del self.root
        self.root = new_root


class MemoryNode():
    def __init__(self, state, h, reward, solved, tree, parent=None, action=None):
        self.state = state
        self.h = h
        self.reward = reward
        self.solved = solved
        self.children = [None for i in range(tree.n_children)]
        self.parent = parent
        self.action = action
        self.tree = tree

    def get_child(self, action):
        return self.children[action.int()]

    def set_child(self, action, state, h, reward, solved):
        self.children[action.int()] = MemoryNode(state, h, reward, solved, self.tree, self, action)
        return self.children[action.int()]

    def get_parent(self):
        return self.parent


class AcyclicTree(MemoryTree):
    def __init__(self, n_children):
        super().__init__(n_children)
        self.visited_states = []

    def set_root(self, state, h):
        if self.root is None:
            self.visited_states.append(pickle.dumps(state))
            self.root = AcyclicNode(state, h, 0, False, self)
        else:
            raise ValueError("Root is already defined")


class AcyclicNode(MemoryNode):
    def __init__(self, state, h, reward, solved, tree, parent=None, action=None):
        self.state = state
        self.h = h
        self.reward = reward
        self.solved = solved
        self.moves = [i for i in range(tree.n_children)]
        self.children = [None for i in range(tree.n_children)]
        self.parent = parent
        self.action = action
        self.tree = tree

    def set_child(self, action, state, h, reward, solved):
        self.moves.remove(action.int())
        if not self.moves:
            self.solved = True
        temp = pickle.dumps(state)
        if temp in self.tree.visited_states:
            return None
        self.children[action.int()] = AcyclicNode(state, h, reward, solved, self.tree, self, action)
        return self.children[action.int()]

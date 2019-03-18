class MemoryTree():
    def __init__(self, n_children):
        self.n_children = n_children
        self.root = None

    def set_root(self, state, h):
        if self.root is not None:
            self.root = MemoryError(state, h, 0, self)
        else:
            raise ValueError("Root is already defined")

    def cut_tree(self, root_action):
        new_root = self.root.children[root_action]
        for k in reversed(range(self.n_children)):
            if k != root_action:
                del self.root.children[k]
        new_root.parent = None
        del self.root
        self.root = new_root


class MemoryNode():
    def __init__(self, state, h, reward, tree, parent=None):
        self.state = state
        self.h = h
        self.reward = reward
        self.children = [None for i in range(tree.n_children)]
        self.parent = parent

    def get_children(self, action):
        return self.children[action]

    def set_child(self, action, state, h, reward):
        self.children[action] = MemoryNode(state, h, reward, self.tree, self)
        return self.children[action]

    def get_parent(self):
        return self.parent

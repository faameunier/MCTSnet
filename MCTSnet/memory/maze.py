import pickle
from .interface import *


class Maze(MemoryManager):
    """Maze memory

    A directed graph that
    represents all known memories

    This is a greedy memory.

    Extends:
        MemoryManager
    """

    def __init__(self, n_children):
        self.n_children = n_children
        self.root = None
        self.known_nodes = []
        self.known_hashes = []
        self.path = []

    def set_root(self, state, h):
        """Set current root

        Set the graph starting point

        Arguments:
            state {torch.tensor} -- State of the environment
            h {torch.tensor} -- Embedding of the state

        Raises:
            ValueError -- Root is already defined
        """
        if self.root is None:
            self.root = MazeNode(state, h, 0, False, self)
            self.known_nodes.append(self.root)
            self.known_hashes.append(pickle.dumps(self.root.state.detach().cpu().numpy()))
        else:
            raise ValueError("Root is already defined")

    def get_root(self):
        """Get root

        Return the current root of the tree.

        Returns:
            MemoryNode -- Root of the tree
        """
        return self.root

    def cut_tree(self, root_action):
        """Cut the tree

        Cut the tree to keep only on the
        root's children. The selected child
        becomes the new root and all other
        children are deleted (and there children
        recursively).

        Arguments:
            root_action {int} -- Child to keep

        Raises:
            MemoryError -- The selected child isn't explored
        """
        self.root = self.root.get_child(root_action)
        self.path = []

    def simulation_reset(self):
        """Reset path

        Reset the last explored path.
        """
        self.path = []  # Security, this should be useless

    def append_to_path(self, node):
        self.path.append(node)

    def get_last_visited(self):
        try:
            return self.path.pop()
        except IndexError:
            return None


class MazeNode(Memory):
    """Maze Node

    A node for a graph memory

    Extends:
        Memory
    """

    def __init__(self, state, h, reward, solved, tree, parent=None, action=None):
        self.state = state
        self.h = h
        self.reward = reward
        self.solved = solved
        self.children = [None for i in range(tree.n_children)]
        # self.parent = parent
        self.action = action
        self.tree = tree

    def get_child(self, action):
        """Get child

        Get child that results from action.

        Arguments:
            action {int} -- action performed

        Returns:
            MazeNode -- resulting node from action
        """
        child = self.children[action.int()]
        if child is not None:
            self.tree.append_to_path(self)
        return child

    def set_child(self, action, state, h, reward, solved):
        """Add a chil to the current Node

        Add a child to the current Node.

        Arguments:
            action {int} -- the action that led to this Node
            state {torch.Tensor} -- State of the new node
            h {torch.Tensor} -- embedding of the new node
            reward {float} -- Reward obtained at this node
            solved {boolean} -- Wether the game is finished or not

        Returns:
            MemoryNode -- The built node
        """
        temp = pickle.dumps(state.detach().cpu().numpy())
        new = False
        try:
            k = self.tree.known_hashes.index(temp)
            self.children[action.int()] = self.tree.known_nodes[k]
        except ValueError:
            new = True
            new_node = MazeNode(state, h, reward, solved, self.tree, None, action)
            self.children[action.int()] = new_node
            self.tree.known_hashes.append(temp)
            self.tree.known_nodes.append(new_node)
        self.tree.append_to_path(self)
        return self.children[action.int()], new

    def get_parent(self):
        """Get Parent

        Return the node's parent

        Returns:
            MemoryNode -- current node's parent
        """
        return self.tree.get_last_visited()

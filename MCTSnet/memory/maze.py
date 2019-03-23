import pickle
from .interface import *


class Maze(MemoryManager):
    """Maze memory

    A directed graph that
    represents all known memories

    Extends:
        MemoryManager
    """

    def __init__(self, n_children):
        self.n_children = n_children
        self.root = None
        self.known_nodes = []
        self.known_hashes = []
        self.path = [self.root]

    def set_root(self, state, h):
        """Set current root

        Set the graph starting point

        Arguments:
            state {torch.tensor or np.array} -- State of the environment
            h {torch.tensor} -- Embedding of the state

        Raises:
            ValueError -- Root is already defined
        """
        if self.root is None:
            self.root = MazeNode(state, h, 0, False, self)
            self.known_nodes.append(self.root)
            self.known_hashes.append(pickle.dumps(self.root))
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
        self.path = [self.root]

    def add_to_path(self, node):
        self.path.append(node)


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
        return self.children[action.int()]

    def set_child(self, action, state, h, reward, solved):
        """Add a chil to the current Node

        Add a child to the current Node.

        Arguments:
            action {int} -- the action that led to this Node
            state {np.array or torch.Tensor} -- State of the new node
            h {torch.Tensor} -- embedding of the new node
            reward {float} -- Reward obtained at this node
            solved {boolean} -- Wether the game is finished or not

        Returns:
            MemoryNode -- The built node
        """
        temp = pickle.dumps(state)
        try:
            k = self.tree.known_hashes.index(temp)
            self.children[action.int()] = self.tree.known_nodes[k]
        except ValueError:
            new_node = MazeNode(state, h, reward, solved, self.tree, self, action)
            self.children[action.int()] = new_node
            self.tree.known_hashes.append(temp)
            self.tree.known_nodes.append(new_node)
        return self.children[action.int()]

    def get_parent(self):
        """Get Parent

        Return the node's parent

        Returns:
            MemoryNode -- current node's parent
        """
        # return self.parent

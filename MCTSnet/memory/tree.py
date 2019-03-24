import pickle
from .interface import *


class MemoryTree(MemoryManager):
    """Memory Tree

    A basic memory tree.

    Extends:
        MemoryManager
    """

    def __init__(self, n_children):
        self.n_children = n_children
        self.root = None

    def set_root(self, state, h):
        """Set tree's root

        Set the tree's root given a state
        and embedding.

        Arguments:
            state {torch.tensor or np.array} -- State of the environment
            h {torch.tensor} -- Embedding of the state

        Raises:
            ValueError -- Root is already defined
        """
        if self.root is None:
            self.root = MemoryNode(state, h, 0, False, self)
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
        new_root = self.root.children[root_action]
        for k in reversed(range(self.n_children)):
            if k != root_action:
                del self.root.children[k]
        if new_root is not None:
            # What if the new_root is None ?
            new_root.parent = None
        else:
            raise MemoryError("New root is None")
        del self.root
        self.root = new_root


class MemoryNode(Memory):
    """Memory Node

    An MemoryNode to build an MemoryTree.

    Extends:
        Memory
    """

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

    def get_child(self, action):
        """Get child

        Get child that results from action.

        Arguments:
            action {int} -- action performed

        Returns:
            MemoryNode -- resulting node from action
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
        self.children[action.int()] = MemoryNode(state, h, reward, solved, self.tree, self, action)
        return self.children[action.int()], True

    def get_parent(self):
        """Get Parent

        Return the node's parent

        Returns:
            MemoryNode -- current node's parent
        """
        return self.parent


class AcyclicTree(MemoryTree):
    """Acyclic Memory Tree

    This Memory Tree keeps track
    of all visited nodes to ensure
    the you do not go through the same
    state twice.

    Extends:
        MemoryTree
    """

    def __init__(self, n_children):
        super().__init__(n_children)
        self.visited_states = []  # All nodes accessibles from root

    def set_root(self, state, h):
        """Set tree's root

        Set the tree's root given a state
        and embedding.

        Arguments:
            state {torch.tensor or np.array} -- State of the environment
            h {torch.tensor} -- Embedding of the state

        Raises:
            ValueError -- Root is already defined
        """
        if self.root is None:
            self.visited_states.append(pickle.dumps(state))
            self.root = AcyclicNode(state, h, 0, False, self)
        else:
            raise ValueError("Root is already defined")

    def cut_tree(self, root_action):
        """Cut the tree

        Cut the tree to keep only on the
        root's children. The selected child
        becomes the new root and all other
        children are deleted (and there children
        recursively). Finally all cycles are
        resetted.

        Arguments:
            root_action {int} -- Child to keep

        Raises:
            MemoryError -- The selected child isn't explored
        """
        new_root = self.root.children[root_action]
        # Cut all branches that but the selected one
        for k in reversed(range(self.n_children)):
            if k != root_action and self.root.children[k] is not None:
                self.remove_children(self.root.children[k])
        # Define the selected branch as root
        if new_root is not None:
            new_root.parent = None
        else:
            raise MemoryError("New root is None")
        self.reset_cycles(self.get_root())
        del self.root
        self.root = new_root

    def remove_children(self, node):
        """Delete nodes from memory

        Physical memory and list of
        visited_nodes. Maybe one of the deleted node
        is accessible from another path, therefore
        cycles should be checked after deletion.

        Arguments:
            node {AcyclicNode} -- the node to delete
        """
        try:
            self.visited_states.remove(pickle.dumps(node.state))
        except ValueError:
            pass
        for c in node.children:
            if c is not None:
                self.remove_children(c)
        del node

    def reset_cycles(self, node):
        """Reset cycles of all nodes

        Check if a blocked path should be
        unlocked. Ie if a node move has been explored
        (removed from the list of explorable moves)
        but the corresponding children is None, that means
        that the future state was already visited.

        However after cutting the tree, we do not know
        which nodes were visited before, and these ones
        may need to be visited again. Simplest way to solve
        this problem is to reset all blocked paths.

        Arguments:
            node {AcyclicNode} -- a node to reset
        """
        for k in range(self.n_children):
            if node.children[k] is None:
                if k not in node.moves:
                    node.moves.append(k)
            else:
                self.reset_cycles(node.children[k])


class AcyclicNode(MemoryNode):
    """Acyclic Node

    An AcyclicNode to build an AcyclicTree.

    Extends:
        MemoryNode
    """

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
        """Add a chil to the current Node

        Add a child to the current Node iif this new node
        has never been visited before (or rather is accessible
        from the current root). Otherwise None is returned
        and no Node is added to the tree.

        Arguments:
            action {int} -- the action that led to this Node
            state {np.array or torch.Tensor} -- State of the new node
            h {torch.Tensor} -- embedding of the new node
            reward {float} -- Reward obtained at this node
            solved {boolean} -- Wether the game is finished or not

        Returns:
            AcyclicNode -- The built node
        """
        self.moves.remove(action.int())
        temp = pickle.dumps(state)
        if temp in self.tree.visited_states:
            return None
        self.children[action.int()] = AcyclicNode(state, h, reward, solved, self.tree, self, action)
        return self.children[action.int()], True

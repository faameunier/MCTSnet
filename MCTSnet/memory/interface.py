class MemoryManager():
    """Memory Interface

    A basic memory interface.
    """

    def __init__(self, n_children):
        raise NotImplementedError

    def set_root(self, state, h):
        """Set root

        Set a node as starting point

        Arguments:
            state {torch.tensor or np.array} -- State of the environment
            h {torch.tensor} -- Embedding of the state
        """
        raise NotImplementedError

    def get_root(self):
        """Get root

        Return the current root.

        Returns:
            MemoryNode -- Root of the tree
        """
        raise NotImplementedError

    def cut_tree(self, root_action):
        """Cut the tree

        A function that is called when
        replanning is occuring. Any
        replaning logic can go here if required

        Arguments:
            root_action {int} -- Child to keep

        Raises:
            MemoryError -- The selected child isn't explored
        """
        pass

    def simulation_reset(self):
        """New simulation callback

        A function that is called each
        time a simulation is started.

        Any logic can go here.
        """
        pass


class Memory():
    """Memory

    An object to store a memory
    """

    def __init__(self, state, h, reward, solved, tree, parent=None, action=None):
        raise NotImplementedError

    def get_child(self, action):
        """Get child

        Get child that results from action.

        Arguments:
            action {int} -- action performed

        Returns:
            Memory -- resulting node from action
        """
        raise NotImplementedError

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
            Memory -- The built node
            Boolean -- new node or not
        """
        raise NotImplementedError

    def get_parent(self):
        """Get Parent

        Return the node's parent

        Returns:
            Memory -- current node's parent
        """
        raise NotImplementedError

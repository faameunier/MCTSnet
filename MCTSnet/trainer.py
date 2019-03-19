from . import models
import torch


class MCTSnetSokoban():
    def __init__(self, env, feature_space, n_embeddings, n_actions, n_simulations=10):
        self.env = env
        self.feature_space = feature_space
        self.n_embeddings = n_embeddings
        self.n_actions = n_actions
        self.n_simulations = n_simulations
        self.backup = models.backup.BetaMLP(self.n_embeddings)
        self.embedding = models.backup.Epsilon(feature_space[0], feature_space[1:], self.n_embeddings)
        self.policy = models.policy.Pi(self.n_embeddings, self.n_actions)
        self.readout = models.readout.Rho(self.n_embeddings, self.n_actions)
        self.model = models.MCTSnet.MCTSnet(self.env, self.backup, self.embedding, self.policy, self.readout, self.n_simulations, self.n_actions)

    @property
    def model(self):
        return self.__model

    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))

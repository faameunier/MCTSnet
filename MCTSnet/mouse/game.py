import numpy as np
import cv2


class Environment(object):
    def __init__(self, grid_size=10, max_time=500, temperature=0.1):
        grid_size = grid_size
        self.grid_size = grid_size
        self.max_time = max_time
        self.temperature = temperature

        # board on which one plays
        self.board = np.zeros((grid_size, grid_size))
        self.position = np.zeros((grid_size, grid_size))

        # coordinate of the cat
        self.x = 0
        self.y = 1

        # self time
        self.t = 0
        self.score = 0
        self.scale = 16

    def get_frame(self):
        b = np.zeros((self.grid_size, self.grid_size, 3)) + 128
        b[self.board > 0, 0] = 256
        b[self.board < 0, 2] = 256
        b[self.x, self.y, :] = 256

        b = cv2.resize(b, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)

        return b

    def step(self, action):
        """This function returns the new state, reward and decides if the
        game ends."""

        self.position = np.zeros((self.grid_size, self.grid_size))

        self.position[self.x, self.y] = 1
        if action == 0:
            if self.x == self.grid_size - 1:
                self.x = self.x - 1
            else:
                self.x = self.x + 1
        elif action == 1:
            if self.x == 0:
                self.x = self.x + 1
            else:
                self.x = self.x - 1
        elif action == 2:
            if self.y == self.grid_size - 1:
                self.y = self.y - 1
            else:
                self.y = self.y + 1
        elif action == 3:
            if self.y == 0:
                self.y = self.y + 1
            else:
                self.y = self.y - 1
        else:
            RuntimeError('Error: action not recognized')

        self.t = self.t + 1
        reward = self.board[self.x, self.y]
        self.board[self.x, self.y] = 0
        game_over = self.t > self.max_time or len(np.argwhere(self.board >= 0.5)) == 0
        if len(np.argwhere(self.board >= 0.01)) == 0:
            reward += 1000
        state = np.array([self.board, self.position])
        self.score += reward
        return state, reward, game_over, None

    def reset(self):
        """This function resets the game and returns the initial state"""

        self.x = np.random.randint(0, self.grid_size - 1, size=1)[0]
        self.y = np.random.randint(0, self.grid_size - 1, size=1)[0]
        self.score = 0
        bonus = 10.0 * np.random.binomial(1, self.temperature, size=self.grid_size**2)
        bonus = bonus.reshape(self.grid_size, self.grid_size)

        malus = -20.0 * np.random.binomial(1, self.temperature, size=self.grid_size**2)
        malus = malus.reshape(self.grid_size, self.grid_size)

        malus[bonus > 0] = 0

        self.board = bonus + malus

        self.position = np.zeros((self.grid_size, self.grid_size))

        self.board[self.x, self.y] = 0
        self.position[self.x, self.y] = 1
        self.t = 0

        state = np.array([self.board, self.position])
        return state


class EnvironmentExploring(Environment):
    def __init__(self, grid_size=10, max_time=500, temperature=0.1):
        super().__init__(grid_size, max_time, temperature)
        self.malus_position = np.zeros((self.grid_size, self.grid_size))

    def step(self, action):
        """This function returns the new state, reward and decides if the
        game ends."""
        state, reward, game_over, _ = super().step(action)
        reward -= 1
        self.score -= 1
        # if reward == 0:
        #     reward -= self.malus_position[self.x, self.y]
        #     self.score -= self.malus_position[self.x, self.y]
        #     self.malus_position[self.x, self.y] += 1
        return state, reward, game_over, None

    def reset(self):
        """This function resets the game and returns the initial state"""
        self.malus_position = np.zeros((self.grid_size, self.grid_size))
        state = super().reset()
        # print(np.concatenate(self.create_state(state), axis=2))
        return state  # self.create_state(state)

    def create_state(self, super_state):
        return np.array((super_state[0], super_state[1], self.malus_position))

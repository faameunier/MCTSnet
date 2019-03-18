from gym_sokoban.envs.sokoban_env import SokobanEnv
from . import utils
import PIL


class SokobanEnvEncoded(SokobanEnv):
    metadata = {'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array']}

    def __init__(self, make_gif=False):
        super().__init__(num_boxes=3, max_steps=200)
        self.make_gif = make_gif
        self.gif = []

    def step(self, action):
        observation, reward, done, info = super().step(action)
        if self.make_gif:
            self.gif.append(self.render(mode="rgb_array"))
        return utils.encode(self.room_state), reward, done, info

    def render(self, mode="rgb_array"):
        img = super().render(mode=mode)
        display_img = PIL.Image.fromarray(img)
        return display_img

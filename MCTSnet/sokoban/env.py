from . import hack
from gym_sokoban.envs.sokoban_env import SokobanEnv
from . import utils
import PIL


class SokobanEnvEncoded(SokobanEnv):
    metadata = {'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array']}

    def __init__(self, make_gif=False):
        super().__init__(num_boxes=3, max_steps=500)
        self.make_gif = make_gif
        self.gif = []
        self.solution = hack.solution.copy()
        trans = {
            0: 1,
            1: 0,
            2: 3,
            3: 2,
            4: 5,
            5: 4,
            6: 7,
            7: 6
        }
        self.solution = [trans[s] for s in self.solution]

    def step(self, action):
        observation, reward, done, info = super().step(action)
        if self.make_gif:
            self.gif.append(self.render(mode="rgb_array"))
        return utils.encode(self.room_state), reward, done, info

    def render(self, mode="rgb_array"):
        img = super().render(mode=mode)
        display_img = PIL.Image.fromarray(img)
        return display_img

    def get_frame(self):
        img = super().render(mode="rgb_array")
        return img

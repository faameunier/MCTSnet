import PIL
import numpy as np
import copy
from gym_sokoban.envs.render_utils import room_to_rgb


def get_image(env):
    img = env.render(mode='rgb_array')
    display_img = PIL.Image.fromarray(img)
    return display_img


def encode(x):
    y = np.zeros((4, 10, 10), dtype=np.float32)
    y[0] = x == 0  # wall
    y[1] = x == 5  # agent
    y[2] = (x == 3) | (x == 4)  # box
    y[3] = x == 2  # target
    return y


def decode(x):
    return ((1 - x[0]) + x[1] * 5 + x[2] * 3 + x[3] * 2).astype(np.uint8)


def state_to_image(state):
    return PIL.Image.fromarray(room_to_rgb(decode(state)))

import PIL
import numpy as np


def get_image(env):
    img = env.render(mode='rgb_array')
    display_img = PIL.Image.fromarray(img)
    return display_img


def encode(x):
    y = np.zeros((4, 10, 10), dtype=np.int8)
    y[0] = x == 0  # wall
    y[1] = x == 5  # agent
    y[2] = (x == 3) | (x == 4)  # box
    y[3] = x == 2  # target
    return y

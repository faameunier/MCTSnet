import PIL
import numpy as np


def get_image(env):
    img = env.render(mode='rgb_array')
    display_img = PIL.Image.fromarray(img)
    return display_img


def encode(x):
    y = np.zeros((10, 10, 4), dtype=np.int8)
    convert = {
        0: [1, 0, 0, 0],
        1: [0, 0, 0, 0],
        5: [0, 1, 0, 0],
        3: [0, 0, 1, 0],
        4: [0, 0, 1, 0],
        2: [0, 0, 0, 1]
    }
    for i in range(10):
        for j in range(10):
            y[i, j] = convert[x[i, j]]
    return y

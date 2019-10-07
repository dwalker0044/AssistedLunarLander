import matplotlib.pyplot as plt
import numpy as np


def plot_map(data):
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(data, aspect="auto")
    cb = plt.colorbar(im)

    ax.set_title("QTable")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(["Nop", "Left", "Main", "Right"])

    plt.draw()

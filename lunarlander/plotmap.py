import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# sphinx_gallery_thumbnail_number = 2

def plot_map(data, shape):
    nop = data[:, 0].reshape(shape)
    left = data[:, 1].reshape(shape)
    main = data[:, 2].reshape(shape)
    right = data[:, 3].reshape(shape)

    fig, axes = plt.subplots(2, 2)

    axes[0, 0].imshow(nop)
    axes[0, 0].set_title("Nop")

    axes[0, 1].imshow(left)
    axes[0, 1].set_title("Left")

    axes[1, 0].imshow(main)
    axes[1, 0].set_title("Main")

    axes[1, 1].imshow(right)
    axes[1, 1].set_title("Right")

    # # We want to show all ticks...
    # ax.set_xticks(np.arange(len(farmers)))
    # ax.set_yticks(np.arange(len(vegetables)))
    # # ... and label them with the respective list entries
    # ax.set_xticklabels(farmers)
    # ax.set_yticklabels(vegetables)
    #
    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")
    #
    # # Loop over data dimensions and create text annotations.
    # for i in range(len(vegetables)):
    #     for j in range(len(farmers)):
    #         text = ax.text(j, i, harvest[i, j],
    #                        ha="center", va="center", color="w")

    # fig.set_title("Q-Learning results")
    fig.tight_layout()
    plt.draw()

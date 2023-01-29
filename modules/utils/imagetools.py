import os

import matplotlib.pyplot as plt


def save_fig(
    fig_id, image_path, tight_layout=True, fig_extension="png", resolution=None
):
    path = os.path.join(image_path, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    if resolution != None:
        plt.savefig(path, format=fig_extension, dpi=resolution)
    else:
        plt.savefig(path, format=fig_extension)

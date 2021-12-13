from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from config import SAVE_PATTERN_PATH


def load_pattern(sentence_index, folder, layer, head):
    file_name = f"layer_{layer}_head_{head}.npy"
    load_path_npy = SAVE_PATTERN_PATH / folder / f'index_{sentence_index}' / 'npy'
    print("loading ", load_path_npy / file_name)
    pattern = np.load(load_path_npy / file_name)
    return pattern


def save_map(map, x_labels, y_labels, path, title=''):
    fig, ax = plt.subplots()
    ax.imshow(map)

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))

    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    ax.set_title(title)

    # fig.tight_layout()
    plt.savefig(path)
    plt.close(fig)

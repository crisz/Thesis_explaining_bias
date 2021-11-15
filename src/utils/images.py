from pathlib import Path

import numpy as np


def load_pattern(sentence_index, layer, head):
    file_name = "pattern_sentence_{}_layer_{}_head_{}".format(sentence_index, layer, head)
    load_path_npy = Path('..') / 'patterns_npy' / (file_name + '.npy')

    pattern = np.load(load_path_npy)
    return pattern
import argparse

import numpy as np
import torch
from sklearn.cluster import KMeans

from model.pattern_classifier import PatternClassifier
from utils.images import load_pattern, get_pattern_classifier, increase_contrast


def main(args):
    pattern_id = args.pattern_id
    method = args.method

    patterns = []
    for layer_index in range(12):
        for head_index in range(12):
            patterns.append(load_pattern(pattern_id, method, layer_index, head_index))
    dim = patterns[0].shape[0]
    tokens = np.zeros((dim,))

    for i, pattern in enumerate(patterns):
        x = increase_contrast(pattern)
        for index_column in range(dim):
            mat = np.zeros((dim, dim))
            mat[:, index_column] = 1.
            result = np.multiply(mat, x).sum()
            if result/dim > 0.2:
                tokens[index_column] += 1
                # if index_column == 12:
                #     print(i//12, i%12)
    print(f"Tokens values for pattern {pattern_id} is: ")
    # print('\t'.join([str(x) for x in tokens]))
    print(tokens)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern-id',
                        help='Pattern id',
                        default=None,
                        type=int)

    parser.add_argument('--method',
                        help='Type of attention',
                        default='normal',
                        type=str)  # TODO restrict choices

    args = parser.parse_args()
    main(args)

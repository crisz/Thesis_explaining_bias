import numpy as np
import torch
from sklearn.cluster import KMeans

from model.pattern_classifier import PatternClassifier
from utils.images import load_pattern


def main():
    patterns = []
    for layer_index in range(12):
        for head_index in range(12):
            patterns.append(load_pattern(180, 'attention', layer_index, head_index)[:7, :7])

    pc = PatternClassifier(dim=7, num_classes=4)
    pc.fake_train()
    categories = ['diagonal', 'vertical', 'block', 'heterogeneous']
    for i, pattern in enumerate(patterns):
        x = np.expand_dims(np.expand_dims(pattern, axis=0), axis=0)
        x = torch.from_numpy(x)
        out = pc.forward(x).detach().numpy()
        category_index = np.argmax(out)
        print(f">> index {i} ({i//12} {i%12}) has category {categories[category_index]}")


if __name__ == '__main__':
    main()

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms
from matplotlib import pyplot as plt

from config import SAVE_PATTERN_PATH
from model.pattern_classifier import RandomShape
from utils.cuda import get_device


def load_pattern(sentence_index, folder, layer, head):
    file_name = f"layer_{layer}_head_{head}.npy"
    load_path_npy = SAVE_PATTERN_PATH / folder / f'index_{sentence_index}' / 'npy'
    print("loading ", load_path_npy / file_name)
    pattern = np.load(load_path_npy / file_name)
    return pattern


def save_map(map, x_labels, y_labels, path, title='', show=False):
    fig, ax = plt.subplots()
    ax.imshow(map)

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))

    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    ax.set_title(title)

    # fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(path)

    plt.close(fig)


def increase_contrast(npy_img):
    img = torch.from_numpy(np.expand_dims(npy_img, axis=0))
    img = torchvision.transforms.ColorJitter(contrast=5).forward(img).numpy()
    img = 0.5 + img - np.min(img) / (np.max(img) - np.min(img))
    img = img.astype(int)
    # save_map(img[0], [], [], None, '', True)
    return img[0]


def get_pattern_classifier(dim=7, epochs=500):
    resnet18 = torchvision.models.resnet18()
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = torch.nn.Linear(num_ftrs, 4)
    resnet18 = resnet18.to(get_device())
    resnet18.double()
    resnet18.train()

    optimizer = torch.optim.Adam(resnet18.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    rs = RandomShape(dim)
    for epoch in range(epochs):
        resnet18.zero_grad()
        optimizer.zero_grad()
        categs = np.array(
            [rs.random_diagonal(), rs.random_vertical(), rs.random_block(), rs.random_heterogeneous()])
        categs = categs.repeat(3, axis=1)

        labels = np.identity(4)
        indices = np.array(list(range(4)))
        np.random.shuffle(indices)
        categs = categs[indices]
        labels = labels[indices]
        inp = torch.from_numpy(categs)
        out = resnet18.forward(inp)
        target = torch.from_numpy(labels)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print(f"Loss at epoch {epoch} is {loss.item()}")
    resnet18.eval()
    return resnet18


if __name__ == '__main__':
    test_img_path = Path('..') / SAVE_PATTERN_PATH / 'attention' / 'index_972' / 'npy' / f'layer_3_head_11.npy'
    test_img = np.load(test_img_path)
    test_img = increase_contrast(npy_img=test_img)
    save_map(test_img, [], [], None, '', True)


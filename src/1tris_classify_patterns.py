import numpy as np
import torch.nn.functional as F
import torch
from matplotlib import pyplot as plt

from utils.images import load_pattern


def main():
    for layer_index in range(2, 6):
        for head_index in range(6, 10):
            pattern = load_pattern(5, layer_index, head_index)

            diagonal_kernel = torch.tensor([[1., 0.5, 0.],
                                            [0.5, 1., 0.5],
                                            [0., 0.5, 1.]])
            diagonal_classification_kernel = torch.tensor([[1., 0.],
                                                           [0., 1]])

            vertical_kernel = torch.tensor([[0., 0.25, 1.],
                                            [0., 0.25, 1.],
                                            [0., 0.25, 1.]])
            vertical_classification_kernel = torch.tensor([[0.25, 1.],
                                                           [0.25, 1.]])

            block_kernel = torch.tensor([[1., 1., 0.],
                                        [1., 1., 0.],
                                        [0., 0., 0.]])
            block_classification_kernel = torch.tensor([[1., 0.],
                                                        [0., 0.]])

            kernels = [diagonal_kernel, vertical_kernel, block_kernel]
            classification_kernels = [diagonal_classification_kernel, vertical_classification_kernel, block_classification_kernel]

            dim = pattern.shape[-1]
            x = torch.Tensor(pattern)
            x = x.view(1, 1, dim, dim)

            for index, kernel in enumerate(kernels):
                output_dim = dim
                kernel = kernel.view(1, 1, 3, 3) / torch.sum(kernel)
                output = x
                while output_dim >= 3:
                    output = F.conv2d(output, kernel, padding=(2, 2), stride=(3, 3))
                    output_dim = output.shape[-1]

                if output_dim == 2:
                    classification_kernel = classification_kernels[index]
                    classification_kernel = classification_kernel.view(1, 1, 2, 2) / torch.sum(classification_kernel)
                    output = F.conv2d(output, classification_kernel, padding=(0, 0), stride=(2, 2))

                score = output.detach().item() * 100

                type = None
                if index == 0:
                    type = "Diagonal"
                elif index == 1:
                    type = "Vertical"
                elif index == 2:
                    type = "Block"

                print("Score for kernel {} layer {} and head {} is {}".format(type, layer_index, head_index, score))
                if score > 0.5:
                    print("{} {} is {}".format(layer_index, head_index, type))


if __name__ == '__main__':
    main()

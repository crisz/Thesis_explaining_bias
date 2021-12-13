import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from tqdm import tqdm


class PatternClassifier(nn.Module):
    def __init__(self, dim, num_classes):
        super(PatternClassifier, self).__init__()

        torch.use_deterministic_algorithms(True)

        self.dim = dim
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.fc1 = nn.Linear(in_features=dim, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=256)
        self.fc5 = nn.Linear(in_features=256, out_features=256)
        self.fc6 = nn.Linear(in_features=256, out_features=256)

        self.fc7 = nn.Linear(in_features=256*64*dim, out_features=64*dim)
        self.fc8 = nn.Linear(in_features=64*dim, out_features=64)
        self.fc9 = nn.Linear(in_features=64, out_features=num_classes)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

    def forward(self, pattern):
        batch_size, channels, length, _ = pattern.shape
        x = pattern.double()
        x = self.fc1(F.relu(self.conv1(x)))
        x = self.fc2(F.relu(self.conv2(x)))
        x = self.fc3(F.relu(self.conv3(x)))
        x = self.fc4(F.relu(self.conv4(x)))
        x = self.fc5(F.relu(self.conv5(x)))
        x = self.fc6(F.relu(self.conv6(x)))

        x = x.reshape(batch_size, -1)
        x = self.fc7(x)
        x = self.dropout1(x)
        x = self.fc8(x)
        x = self.dropout2(x)
        x = self.fc9(x)

        x = nn.Softmax()(x)
        return x

    def random_diagonal(self):
        mat = torch.from_numpy(np.identity(self.dim))
        mat = mat + np.random.normal(0, 0.1, mat.shape)**2
        return np.expand_dims(mat, axis=0)

    def random_vertical(self):
        mat = np.zeros((self.dim, self.dim))
        column = random.randrange(0, self.dim)
        mat[:, column] = 1
        mat = mat + np.random.normal(0, 0.1, mat.shape)**2
        return np.expand_dims(mat, axis=0)

    def random_block(self):
        mat = np.zeros((self.dim, self.dim))
        row = random.randrange(0, self.dim)
        column = random.randrange(0, self.dim)
        mat[:row, :column] = 1
        mat[row:, column:] = 1
        mat = mat + np.random.normal(0, 0.1, mat.shape)**2
        return np.expand_dims(mat, axis=0)

    def random_heterogeneous(self):
        mat = np.zeros((self.dim, self.dim))
        mat = mat + np.random.normal(0, 0.6, mat.shape)**2
        return np.expand_dims(mat, axis=0)

    def fake_train(self, epochs=200):
        print(">> Fake train started")
        self.train()
        self.double()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        scheduler = MultiStepLR(optimizer, milestones=(80, 120, 150), gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss()
        for seed in (42,):
            np.random.seed(seed)
            torch.random.manual_seed(seed)
            for epoch in range(epochs):
                self.zero_grad()
                optimizer.zero_grad()
                categs = np.array(
                    [self.random_diagonal(), self.random_vertical(), self.random_block(), self.random_heterogeneous()])
                labels = np.identity(4)
                indices = np.array(list(range(4)))
                np.random.shuffle(indices)
                categs = categs[indices]
                labels = labels[indices]
                inp = torch.from_numpy(categs)
                out = self.forward(inp)
                target = torch.from_numpy(labels)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
                scheduler.step()
                print(f"Loss at epoch {epoch} is {loss.item()}")




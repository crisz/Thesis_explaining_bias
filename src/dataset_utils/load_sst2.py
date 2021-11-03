import numpy as np
import pandas as pd
from config import DATASET_PATH_BASE

DEV_PATH = DATASET_PATH_BASE / 'sst2' / 'dev.tsv'
TRAIN_PATH = DATASET_PATH_BASE / 'sst2' / 'train.tsv'
TEST_PATH = DATASET_PATH_BASE / 'sst2' / 'test.tsv'


def load_sst2_dataset(path):
    df = pd.read_csv(path, sep='\t', header=None, names=['similarity', 's1'])
    npy = df.values
    dataset = np.swapaxes(npy, 0, 1)
    labels, data = dataset
    labels = labels.astype(np.int)
    labels = np.array(labels).reshape(-1, 1)
    return labels, data


def load_sst2_train_dataset():
    return load_sst2_dataset(TRAIN_PATH)


def load_sst2_val_dataset():
    return load_sst2_dataset(TEST_PATH)

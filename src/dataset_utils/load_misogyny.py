import numpy as np
import pandas as pd
from config import DATASET_PATH_BASE

DEV_PATH = DATASET_PATH_BASE / 'misogyny_EN' / 'miso_dev.tsv'
TRAIN_PATH = DATASET_PATH_BASE / 'misogyny_EN' / 'miso_train.tsv'
TEST_PATH = DATASET_PATH_BASE / 'misogyny_EN' / 'miso_test.tsv'


def load_misogyny_dataset(path):
    df = pd.read_csv(path, sep='\t')[['text', 'misogynous']]
    npy = df.values
    dataset = np.swapaxes(npy, 0, 1)
    data, labels = dataset
    labels = labels.astype(np.int)
    labels = np.array(labels).reshape(-1, 1)
    return labels, data


def load_misogyny_train_dataset():
    return load_misogyny_dataset(TRAIN_PATH)


def load_misogyny_val_dataset():
    return load_misogyny_dataset(TEST_PATH)

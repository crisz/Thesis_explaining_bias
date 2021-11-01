import numpy as np
import pandas as pd
from config import DATASET_PATH_BASE

DEV_PATH = DATASET_PATH_BASE / 'sst2' / 'dev.tsv'
TRAIN_PATH = DATASET_PATH_BASE / 'sst2' / 'train.tsv'
TEST_PATH = DATASET_PATH_BASE / 'sst2' / 'test.tsv'


def load_train_dataset():
    train_df = pd.read_csv(TRAIN_PATH, sep='\t', header=None, names=['similarity', 's1'])
    train_npy = train_df.values
    train_dataset = np.swapaxes(train_npy, 0, 1)
    train_labels, train_data = train_dataset
    return train_labels, train_data

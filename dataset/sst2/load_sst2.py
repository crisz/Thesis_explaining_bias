import pandas as pd
import os
from pathlib import Path


DEV_PATH = Path('.') / 'dev.tsv'
TRAIN_PATH = Path('.') / 'train.tsv'
TEST_PATH = Path('.') / 'test.tsv'


def load_train_dataset():
    train_df = pd.read_csv(TRAIN_PATH, sep='\t', header=None, names=['similarity', 's1'])
    return train_df.values

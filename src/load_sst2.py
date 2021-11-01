import pandas as pd
from config import DATASET_PATH_BASE

DEV_PATH = DATASET_PATH_BASE / 'sst2' / 'dev.tsv'
TRAIN_PATH = DATASET_PATH_BASE / 'sst2' / 'train.tsv'
TEST_PATH = DATASET_PATH_BASE / 'sst2' / 'test.tsv'


def load_train_dataset():
    print(TRAIN_PATH.resolve())
    print(TRAIN_PATH.exists())
    train_df = pd.read_csv(TRAIN_PATH, sep='\t', header=None, names=['similarity', 's1'])
    return train_df.values

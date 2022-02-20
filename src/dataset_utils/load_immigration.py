import numpy as np
import pandas as pd
from config import DATASET_PATH_BASE

DEV_PATH = DATASET_PATH_BASE / 'immigration_EN' / 'mig_en_dev.tsv'
TRAIN_PATH = DATASET_PATH_BASE / 'immigration_EN' / 'mig_en_train.tsv'
TEST_PATH = DATASET_PATH_BASE / 'immigration_EN' / 'mig_en_test.tsv'


def load_immigration_dataset(path):
    df = pd.read_csv(path, sep='\t')[['text', 'HS']]
    df_positive_class = df[df['HS'] == 1]
    df_negative_class = df[df['HS'] == 0]

    print(df_positive_class.describe())
    print(df_negative_class.describe())

    print(df_positive_class.head())
    print(df_negative_class.head())

    undersampled = df_negative_class.sample(len(df_positive_class))
    df = pd.concat([df_positive_class, undersampled], axis=0).sample(frac=1).reset_index(drop=True)
    print(df.describe())

    npy = df.values
    dataset = np.swapaxes(npy, 0, 1)
    data, labels = dataset
    labels = labels.astype(np.int)
    labels = np.array(labels).reshape(-1, 1)
    return labels, data


def load_immigration_train_dataset():
    return load_immigration_dataset(TRAIN_PATH)


def load_immigration_val_dataset():
    return load_immigration_dataset(TEST_PATH)

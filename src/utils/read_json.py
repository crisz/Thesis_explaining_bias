import json

from config import JSON_PATH

JSON_FILE = JSON_PATH / 'val_selected_indices.json'


def load_ids(path=None, dataset='misog'):
    if dataset == 'misog':
        path = path if path is not None else JSON_FILE
        with open(path, 'r') as f:
            selected = json.load(f)['selected']
            selected = list(map(lambda x: x['index'], selected))
            return selected
    else: return [1219, 871, 729, 1138, 297]


def load_fp_ids(path=None):
    path = path if path is not None else JSON_FILE
    with open(path, 'r') as f:
        selected = json.load(f)['selected']
        selected = filter(lambda x: x['type'] == 'FP', selected)
        selected = list(map(lambda x: x['index'], selected))
        return selected


def load_fn_ids(path=None):
    path = path if path is not None else JSON_FILE
    with open(path, 'r') as f:
        selected = json.load(f)['selected']
        selected = filter(lambda x: x['type'] == 'FN', selected)
        selected = list(map(lambda x: x['index'], selected))
        return selected


if __name__ == '__main__':
    print(load_ids())
    print(load_fp_ids())
    print(load_fn_ids())

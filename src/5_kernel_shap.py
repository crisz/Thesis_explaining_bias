import logging
from pathlib import Path

import numpy as np

from dataset_utils.load_misogyny import load_misogyny_val_dataset, load_misogyny_train_dataset
from explainers.ShapWrapper import ShapWrapper
from utils.cuda import get_device
from utils.save_model import load_model
import shap


def main():
    logging.getLogger("shap").setLevel(logging.WARNING)

    val_labels, val_sentences = load_misogyny_val_dataset()
    model, tokenizer = load_model('bert-base-uncased-fine-tuned-misogyny')
    print(">>> Loaded model")
    device = get_device()
    model.to(device)
    wrapped = ShapWrapper(model=model, tokenizer=tokenizer)
    train_data_size = 100
    test_data_size = 10
    offset = train_data_size + test_data_size
    train_data = np.array(val_sentences[test_data_size:offset]).reshape(-1, 1)
    test_data = np.array(val_sentences[:test_data_size]).reshape(-1, 1)

    print(train_data.shape)
    e = shap.KernelExplainer(wrapped, train_data)
    shap_values = e.shap_values(test_data, nsamples=10)
    print("Shap values length: ", len(shap_values))
    for i, value in enumerate(shap_values):
        print(value.shape)
        np.save(Path('.') / f'shap_values_{i}.npy', value)


if __name__ == '__main__':
    main()

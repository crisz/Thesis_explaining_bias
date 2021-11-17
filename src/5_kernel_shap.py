import logging

import numpy as np
import torch

from dataset_utils.load_misogyny import load_misogyny_val_dataset, load_misogyny_train_dataset
from dataset_utils.load_sst2 import load_sst2_val_dataset
from explainers.ShapWrapper import ShapWrapper
from utils.cuda import get_device
from utils.huggingface import get_tokens_from_sentences
from utils.save_model import load_model
import shap


def main():
    logging.getLogger("shap").setLevel(logging.WARNING)

    val_labels, val_sentences = load_misogyny_val_dataset()
    model, tokenizer = load_model('bert-base-uncased-fine-tuned-misogyny')

    wrapped = ShapWrapper(model=model, tokenizer=tokenizer)
    train_data = np.array(val_sentences[10:]).reshape(-1, 1)
    test_data = np.array(val_sentences[:10]).reshape(-1, 1)

    print(train_data.shape)
    e = shap.KernelExplainer(wrapped, train_data)
    e.shap_values(test_data, nsamples=10)


if __name__ == '__main__':
    main()

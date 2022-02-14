import logging
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_utils.load_misogyny import load_misogyny_val_dataset, load_misogyny_train_dataset
from explainers.KernelShapWrapper import KernelShapWrapper
from utils.cuda import get_device
from utils.save_model import load_model
import shap
from sklearn.preprocessing import OrdinalEncoder


def main():
    logging.getLogger("shap").setLevel(logging.WARNING)

    val_labels, val_sentences = load_misogyny_val_dataset()
    model, tokenizer = load_model('bert-base-uncased-fine-tuned-misogyny')
    print(">>> Loaded model")
    device = get_device()
    model.to(device)

    train_data_size = 100

    indices = [180, 289, 324, 425, 485, 523, 568, 779, 817, 972]

    splitted_train_sentences = [val_sentences[i].split(' ') for i in range(train_data_size)]
    padded_train_data = []
    for sentence in splitted_train_sentences:
        sen_len = len(sentence)
        padding = 64 - sen_len
        padded_sentence = sentence + ['', ]*padding
        padded_sentence = np.array(padded_sentence)
        padded_train_data.append(padded_sentence)

    splitted_test_sentences = [val_sentences[i].split(' ') for i in indices]
    padded_test_data = []
    for sentence in splitted_test_sentences:
        sen_len = len(sentence)
        padding = 64 - sen_len
        padded_sentence = sentence + ['', ]*padding
        padded_sentence = np.array(padded_sentence)
        padded_test_data.append(padded_sentence)

    padded_train_data = np.array(padded_train_data)

    encoder = OrdinalEncoder()
    encoder.fit([*padded_train_data, *padded_test_data])
    padded_train_data = encoder.transform(padded_train_data)
    padded_test_data = encoder.transform(padded_test_data)

    train_data = np.array(padded_train_data).reshape((-1, 64))
    test_data = np.array(padded_test_data).reshape((-1, 64))
    print(train_data.shape)
    print(test_data.shape)
    wrapped = KernelShapWrapper(model=model, tokenizer=tokenizer, encoder=encoder)

    print(train_data.shape)

    df = pd.DataFrame(train_data)
    print(df.head())
    e = shap.KernelExplainer(wrapped, df)

    df2 = pd.DataFrame(test_data)
    print(df2.head())
    shap_values = e.shap_values(X=df2, l1_reg="aic", nsamples="auto")

    np.save('./deep_shap_result.npy', shap_values, allow_pickle=True)

    s0, s1 = shap_values
    np.save(Path('.') / 's0.npy', s0)
    np.save(Path('.') / 's1.npy', s1)

    print(s0.shape)
    print(s1.shape)

    print(s1)

    # for i, data in enumerate(test_data):
    #     print("shape before reshaping: ", data.shape)
    #     print("shape after reshaping: ", test_data[i].reshape(1, -1).shape)
    #     shap_values = e.shap_values(X=test_data[i].reshape(-1, 1), l1_reg="aic", nsamples="auto")
    #     print("Shap values length and type: ", len(shap_values), type(shap_values))
    #     print("Original sentence: ", val_sentences[i])
    #     print("Shap values: ")
    #     print(shap_values)


if __name__ == '__main__':
    main()

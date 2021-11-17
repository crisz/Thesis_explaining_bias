import logging
from pathlib import Path

import numpy as np

from dataset_utils.load_misogyny import load_misogyny_val_dataset, load_misogyny_train_dataset
from explainers.ShapWrapper import ShapWrapper
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
    test_data_size = 10
    offset = train_data_size + test_data_size

    splitted_val_sentences = [sentence.split(' ') for sentence in val_sentences]
    input_sentences = []
    for sentence in splitted_val_sentences:
        sen_len = len(sentence)
        padding = 64 - sen_len
        padded_sentence = sentence + ['', ]*padding
        padded_sentence = np.array(padded_sentence)
        input_sentences.append(padded_sentence)

    input_sentences = np.array(input_sentences)

    encoder = OrdinalEncoder()
    input_sentences = encoder.fit_transform(input_sentences)

    train_data = np.array(input_sentences[test_data_size:offset]).reshape((-1, 64, 1))
    test_data = np.array(input_sentences[:test_data_size]).reshape((-1, 64, 1))

    wrapped = ShapWrapper(model=model, tokenizer=tokenizer, encoder=encoder)

    print(train_data.shape)
    e = shap.KernelExplainer(wrapped, train_data)

    for i, data in enumerate(test_data):
        shap_values = e.shap_values(data.reshape(1, -1), nsamples=10)
        print("Shap values length and type: ", len(shap_values), type(shap_values))
        print("Original sentence: ", val_sentences[i])
        print("Shap values: ")
        print(shap_values)


if __name__ == '__main__':
    main()

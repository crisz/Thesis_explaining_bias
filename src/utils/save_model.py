from transformers import BertTokenizer

from config import MODEL_SAVE_PATH
import os

from model.custom_bert import BertForSequenceClassification


def save_model(model, tokenizer, model_name):
    output_dir = MODEL_SAVE_PATH / model_name

    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def load_model(model_name):
    output_dir = MODEL_SAVE_PATH / model_name
    print("Loading model at path {}".format(output_dir))
    model = BertForSequenceClassification.from_pretrained(output_dir)
    tokenizer = BertTokenizer.from_pretrained(output_dir)
    return model, tokenizer

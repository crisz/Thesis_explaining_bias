from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from dataset_utils.load_misogyny import load_misogyny_val_dataset
from explainers.DeepShapWrapper import DeepShapWrapper
from utils.cuda import get_device
from utils.huggingface import get_tokens_from_sentences
from utils.save_model import load_model
import shap


def main():
    model, tokenizer = load_model('bert-base-uncased-fine-tuned-misogyny')
    val_labels, val_sentences = load_misogyny_val_dataset()
    val_input_ids, val_attention_masks, _ = get_tokens_from_sentences(val_sentences, tokenizer=tokenizer)

    device = get_device()

    model.to(device)
    model.eval()

    indices = [180, 289, 324, 425, 485, 523, 568, 779, 817, 972]

    embedding = torch.empty((973, 64, 768))
    for i in tqdm(range(973)):
        with torch.no_grad():
            out = model.forward(
                val_input_ids[i:i+1].to(device),
                val_attention_masks[i:i+1].to(device),
                token_type_ids=None,
                labels=None,
                return_dict=True,
                output_attentions=True
            )
        embedding[i] = out.embedding_outputs[0]

    wrapped = DeepShapWrapper(model=model)

    test_embeddings = embedding[indices].to(device)  # [embedding[i] for i in indices]
    test_ids = val_input_ids[indices].to(device)  # [val_input_ids[i] for i in indices]

    e = shap.DeepExplainer(wrapped, embedding[:200])
    shap_values = e.shap_values(test_embeddings)

    s1 = torch.from_numpy(shap_values[1])
    s1 = torch.sum(s1, dim=2)

    for sentence_index, _ in enumerate(s1):
        for index, token in enumerate(test_ids[sentence_index]):
            decoded_token = tokenizer.decode(token)
            print("{}, ".format(''.join(decoded_token.split(' '))), end=' ')
        print()
        for index, token in enumerate(test_ids[sentence_index]):
            print("{}, ".format(s1[sentence_index, index].item()), end=' ')
        print()
        print()


if __name__ == '__main__':
    main()

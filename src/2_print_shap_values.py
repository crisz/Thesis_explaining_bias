from pathlib import Path

import numpy as np
import torch

from dataset_utils.load_misogyny import load_misogyny_val_dataset
from explainers.DeepShapWrapper import DeepShapWrapper
from utils.cuda import get_device
from utils.huggingface import get_tokens_from_sentences
from utils.save_model import load_model
import shap


def main():
    model, tokenizer = load_model('bert-base-uncased-fine-tuned-misogyny')
    val_labels, val_sentences = load_misogyny_val_dataset()
    val_sentences = val_sentences[:64]
    val_input_ids, val_attention_masks, _ = get_tokens_from_sentences(val_sentences, tokenizer=tokenizer)

    device = get_device()

    model.to(device)
    model.eval()

    # sentence_index = 127

    with torch.no_grad():
        out = model.forward(
            val_input_ids.to(device),
            val_attention_masks.to(device),
            token_type_ids=None,
            labels=None,
            return_dict=True,
            output_attentions=True
        )

    wrapped = DeepShapWrapper(model=model)

    embedding = out.embedding_outputs
    e = shap.DeepExplainer(wrapped, embedding[10:])
    shap_values = e.shap_values(embedding[9:10])
    for i, shap_value in enumerate(shap_values):
        np.save(Path('.') / f'deep_shap_value_{i}.npy', shap_value)

    s1 = torch.from_numpy(shap_values[1])
    token_len = list(val_attention_masks[9]).index(0)
    # s1 = s1[:, :token_len, :]
    s1 = torch.sum(s1, dim=2)

    for index, token in enumerate(val_input_ids[9]):
        # if index == token_len:
        #     break

        decoded_token = tokenizer.decode(token)
        print("{}({:.2f})".format(decoded_token, s1[0, index].item()), end=' ')


if __name__ == '__main__':
    main()

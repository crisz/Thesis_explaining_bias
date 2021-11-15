from pathlib import Path

import numpy as np

from dataset_utils.load_misogyny import load_misogyny_val_dataset
from utils.cuda import get_device
from utils.huggingface import get_tokens_from_sentences
from utils.save_model import load_model
import matplotlib.pyplot as plt


# Nota: il modello è molto sensibile al soggetto (I/you)
def main():
    model, tokenizer = load_model('bert-base-uncased-fine-tuned-misogyny')
    val_labels, val_sentences = load_misogyny_val_dataset()
    val_input_ids, val_attention_masks, _ = get_tokens_from_sentences(val_sentences, tokenizer=tokenizer)

    device = get_device()

    model.to(device)
    model.eval()

    sentence_index = 5
    current_sentence_index = 0

    out = model.forward(
        val_input_ids[sentence_index:(sentence_index+1)].to(device),
        val_attention_masks[sentence_index:(sentence_index+1)].to(device),
        token_type_ids=None,
        labels=None,
        return_dict=True,
        output_attentions=True
    )

    for current_layer in range(12):
        for current_head in range(12):
            current_attention_map = out.attentions[current_layer][current_sentence_index, current_head].detach().numpy()
            decoded_tokens = [tokenizer.decode(token) for token in val_input_ids[sentence_index]]
            real_sentence_length = decoded_tokens.index("[ P A D ]")
            decoded_tokens = decoded_tokens[:real_sentence_length]
            current_attention_map = current_attention_map[:real_sentence_length, :real_sentence_length]

            fig, ax = plt.subplots()
            ax.imshow(current_attention_map)
            ax.set_xticks(np.arange(len(decoded_tokens)))
            ax.set_yticks(np.arange(len(decoded_tokens)))
            ax.set_xticklabels(decoded_tokens)
            ax.set_yticklabels(decoded_tokens)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")

            ax.set_title("Attention map for layer {} and head {} ".format(current_layer, current_head))
            fig.tight_layout()
            file_name = "pattern_sentence_{}_layer_{}_head_{}".format(sentence_index, current_layer, current_head)
            save_path = Path('..') / 'patterns' / (file_name + '.png')
            # plt.savefig(save_path)
            plt.close(fig)

            save_path_npy = Path('..') / 'patterns_npy' / (file_name + '.npy')
            np.save(save_path_npy, current_attention_map)


if __name__ == '__main__':
    main()

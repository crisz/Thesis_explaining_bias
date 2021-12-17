import numpy as np
import torch
from matplotlib import pyplot as plt

from dataset_utils.load_misogyny import load_misogyny_val_dataset
from dataset_utils.load_sst2 import load_sst2_val_dataset
from utils.cuda import get_device
from utils.huggingface import get_tokens_from_sentences
from utils.save_model import load_model


def calculate_token_attribution(hidden_token, embedding):
    grad = torch.autograd.grad(hidden_token, embedding, grad_outputs=torch.ones_like(hidden_token), retain_graph=True)[0]
    grad = torch.norm(grad, dim=2)[0]
    grad = grad / torch.sum(grad)
    return grad


def compute_negative_entropy(
    inputs: tuple, attention_mask: torch.Tensor, return_values=False
):
    """Compute the negative entropy across layers of a network for given inputs.

    Args:
        - input: tuple. Tuple of length num_layers. Each item should be in the form: BHSS
        - attention_mask. Tensor with dim: BS
    """
    inputs = torch.stack(inputs)  #  LayersBatchHeadsSeqlenSeqlen
    assert inputs.ndim == 5, "Here we expect 5 dimensions in the form LBHSS"

    #  average over attention heads
    pool_heads = inputs.mean(2)

    batch_size = pool_heads.shape[1]
    print("batch_size=", batch_size)
    samples_entropy = list()
    neg_entropies = list()
    for b in range(batch_size):
        #  get inputs from non-padded tokens of the current sample
        mask = attention_mask[b]
        sample = pool_heads[:, b, mask.bool(), :]
        sample = sample[:, :, mask.bool()]
        print("Sample=", sample)

        #  get the negative entropy for each non-padded token
        neg_entropy = (sample.softmax(-1) * sample.log_softmax(-1)).sum(-1)
        if return_values:
            neg_entropies.append(neg_entropy.detach())

        #  get the "average entropy" that traverses the layer
        mean_entropy = neg_entropy.mean(-1)

        #  store the sum across all the layers
        samples_entropy.append(mean_entropy.sum(0))
        print("samples_entropy=", samples_entropy)

    # average over the batch
    final_entropy = torch.stack(samples_entropy).mean()
    print("final**", final_entropy)
    if return_values:
        return final_entropy, neg_entropies
    else:
        return final_entropy


def main():
    model, tokenizer = load_model('bert-base-uncased-fine-tuned-misogyny')
    val_labels, val_sentences = load_misogyny_val_dataset()
    # val_sentences[3] = "My mother is a bitch"
    val_input_ids, val_attention_masks, _ = get_tokens_from_sentences(val_sentences, tokenizer=tokenizer)

    device = get_device()

    model.to(device)
    model.eval()
    model.zero_grad()

    sentence_index = 972
    attention_masks = val_attention_masks[sentence_index:sentence_index+1]
    out = model.forward(
        val_input_ids[sentence_index:sentence_index+1].to(device),
        attention_masks.to(device),
        token_type_ids=None,
        labels=None,
        return_dict=True,
        output_attentions=True
    )

    attention_masks = attention_masks.to('cpu')
    print(len(out.attentions))
    print(attention_masks.shape)
    total_entropy, entropy_attention_map = compute_negative_entropy(
        out.attentions,
        attention_mask=attention_masks,
        return_values=True)

    print([map.shape for map in entropy_attention_map])
    print(len(entropy_attention_map))

    for i, current_attention_map in enumerate(entropy_attention_map):
        # Positive entropy
        current_attention_map = -current_attention_map

        # Removing CLS and SEP
        current_attention_map = current_attention_map[:, 1:-1]

        # Rescaling
        current_attention_map = (1 / current_attention_map).log()

        fig, ax = plt.subplots()

        # **** Riportare
        # La rappresentazione di book a livello 4 viene con poco contesto

        x = current_attention_map.numpy().sum(axis=0)
        print(x)
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        print("Tokens entropy sum: ")
        print(x)
        ax.imshow(current_attention_map)
        decoded_tokens = [tokenizer.decode(token) for token in val_input_ids[i+sentence_index]]
        pad_token = "[ P A D ]"
        real_sentence_length = decoded_tokens.index(pad_token) if pad_token in decoded_tokens else 64
        decoded_tokens = decoded_tokens[:real_sentence_length]
        decoded_tokens = decoded_tokens[1:-1]
        print("decoded tokens len: ", len(decoded_tokens))
        print("current attention map: ", len(current_attention_map))
        ax.set_xticks(np.arange(len(decoded_tokens)))
        ax.set_yticks(np.arange(12))
        ax.set_xticklabels(decoded_tokens)
        ax.set_yticklabels(list(range(12)))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        plt.show()


if __name__ == '__main__':
    main()

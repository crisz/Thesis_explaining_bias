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


def main():
    model, tokenizer = load_model('bert-base-uncased-fine-tuned-misogyny')
    val_labels, val_sentences = load_misogyny_val_dataset()
    # val_sentences[3] = "My mother is a bitch"
    val_input_ids, val_attention_masks, _ = get_tokens_from_sentences(val_sentences, tokenizer=tokenizer)

    device = get_device()

    model.to(device)
    model.eval()
    model.zero_grad()

    sentence_index = 5

    out = model.forward(
        val_input_ids[sentence_index:sentence_index+1].to(device),
        val_attention_masks[sentence_index:sentence_index+1].to(device),
        token_type_ids=None,
        labels=None,
        return_dict=True,
        output_attentions=True
    )

    layer_index = 10
    head_index = 1

    value = out.value[layer_index]
    hidden = out.hidden_weights[layer_index]
    ds = len(val_input_ids[sentence_index])  # =~30
    dv = value.shape[-1]  # = 64
    d = 64  # = 64
    print(hidden.shape)
    attention = out.attentions[layer_index]
    # T = torch.matmul(out.value[layer_index], out.hidden_weights[layer_index])
    T = value
    # Decomposizione
    U, S, V = torch.Tensor.svd(T, some=False, compute_uv=True)
    print("U S, and V shapes are: ", U.shape, S.shape, V.shape)
    bound = torch.finfo(S.dtype).eps * max(U.shape[1], V.shape[1])
    grater_than_bound: torch.Tensor = S > bound
    # noinspection PyArgumentList
    basis_start_index = torch.max(torch.sum(grater_than_bound, dtype=int, axis=2))
    null_space = U[:, :, :, basis_start_index:]
    B = torch.matmul(attention, null_space)
    transpose_B = torch.transpose(B, -1, -2)
    projection_attention = torch.matmul(null_space, transpose_B)
    projection_attention = torch.transpose(projection_attention, -1, -2)
    effective_attention = torch.sub(attention, projection_attention)
    current_attention_map = effective_attention[0, head_index].detach().numpy()

    decoded_tokens = [tokenizer.decode(token) for token in val_input_ids[sentence_index]]
    real_sentence_length = decoded_tokens.index("[ P A D ]")
    current_attention_map = current_attention_map[:real_sentence_length, :real_sentence_length]

    decoded_tokens = decoded_tokens[:real_sentence_length]

    fig, ax = plt.subplots()

    ax.imshow(current_attention_map)

    ax.set_xticks(np.arange(len(decoded_tokens)))
    ax.set_yticks(np.arange(len(decoded_tokens)))
    ax.set_xticklabels(decoded_tokens)
    ax.set_yticklabels(decoded_tokens)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.show()
    # EW = out.value[layer_index] #ds=64; dv=768
    # print("EW.shape", EW.shape)
    # H = out.hidden_weights[layer_index][head_index*64:(head_index+1)*64, :]
    # EW = EW.reshape((12*64, 64)).transpose(1, 0)
    # T = torch.matmul(EW, H)
    # null_space_T = null_space(T.transpose(1, 0).detach().numpy())
    # null_space_T = torch.Tensor(null_space_T).transpose(1, 0)
    # print("nst", null_space_T.shape)
    # attention_transposed = attention.reshape(768, 64)
    # projection = torch.matmul(null_space_T, attention_transposed)
    # print(projection.shape)
    # value = out.value[layer_index]
    # result = torch.matmul(attention, value)
    # result = result.permute(0, 2, 1, 3).contiguous().view(*out.context[1].shape)
    # T = result


if __name__ == '__main__':
    main()

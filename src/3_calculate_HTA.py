import numpy as np
import torch

from dataset_utils.load_misogyny import load_misogyny_val_dataset
from dataset_utils.load_sst2 import load_sst2_val_dataset
from utils.cuda import get_device
from utils.huggingface import get_tokens_from_sentences
from utils.save_model import load_model
import shap


def calculate_token_attribution(hidden_token, embedding):
    grad = torch.autograd.grad(hidden_token, embedding, grad_outputs=torch.ones_like(hidden_token), retain_graph=True)[0]
    grad = torch.norm(grad, dim=2)[0]
    grad = grad / torch.sum(grad)
    return grad


def main():
    model, tokenizer = load_model('bert-base-uncased-fine-tuned-misogyny')
    val_labels, val_sentences = load_misogyny_val_dataset()
    val_sentences[3] = "My mother is a bitch"
    val_input_ids, val_attention_masks, _ = get_tokens_from_sentences(val_sentences, tokenizer=tokenizer)

    device = get_device()

    model.to(device)
    model.eval()
    model.zero_grad()

    sentence_index = 3

    out = model.forward(
        val_input_ids[sentence_index:sentence_index+1].to(device),
        val_attention_masks[sentence_index:sentence_index+1].to(device),
        token_type_ids=None,
        labels=None,
        return_dict=True,
        output_attentions=True
    )

    current_sentence = val_sentences[sentence_index]

    print("Examining the following sentence: ")
    print(current_sentence)
    print()
    print("The ground truth label is: {}".format(val_labels[sentence_index][0]))
    print("The predicted label is: {}".format(np.argmax(out.logits.detach().numpy())))
    print()
    print()

    last_layer_cls = out.hidden_states[12][0]
    hta = calculate_token_attribution(last_layer_cls, out.embedding_outputs)
    print(hta)
    print(hta.shape)

    for index, token in enumerate(val_input_ids[sentence_index]):
        decoded_token = tokenizer.decode(token)
        print("{}({:.2f})".format(decoded_token, hta[index].item()), end=' ')


if __name__ == '__main__':
    main()

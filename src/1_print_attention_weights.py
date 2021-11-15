import numpy as np
import torch

from dataset_utils.load_misogyny import load_misogyny_val_dataset
from dataset_utils.load_sst2 import load_sst2_val_dataset
from utils.cuda import get_device
from utils.huggingface import get_tokens_from_sentences
from utils.save_model import load_model


# Nota: il modello è molto sensibile al soggetto (I/you)
def main():
    model, tokenizer = load_model('bert-base-uncased-fine-tuned-misogyny')
    val_labels, val_sentences = load_misogyny_val_dataset()
    val_sentences = ["Not all women deserve respect"]
    val_labels = [[0]]
    val_input_ids, val_attention_masks, _ = get_tokens_from_sentences(val_sentences, tokenizer=tokenizer)

    device = get_device()

    model.to(device)
    model.eval()

    sentence_index = 0
    token_index = 3

    out = model.forward(
        val_input_ids[sentence_index:sentence_index+1].to(device),
        val_attention_masks[sentence_index:sentence_index+1].to(device),
        token_type_ids=None,
        labels=None,
        return_dict=True,
        output_attentions=True
    )

    # current_sentence = val_sentences[sentence_index]

    print("Number of attentions: ", len(out.attentions))
    # Get the last layer, the final one

    last_layer_attention = out.attentions[3]  # output shape: [sentences_number, heads, length, length]

    print(last_layer_attention.shape)

    # We only have one sentece, let's examine
    # last_layer_current_sentence = last_layer_attention[0]

    # Perform average over heads
    # last_layer_avg = torch.mean(last_layer_current_sentence, dim=0, keepdim=False)  # shape: LxL = 64x64

    # The above is what "BERT dark secrets" paper refers as "self-attention maps"
    #
    # for hidden_state in out.hidden_states:
    #     print("hs", hidden_state.shape)
    #
    # for input_id in val_input_ids:
    #     print("iid", input_id.shape)

    # print("Examining the following sentence: ")
    # print(current_sentence)
    # print()
    # print("The ground truth label is: {}".format(val_labels[sentence_index][0]))
    # prediction = torch.nn.functional.softmax(out.logits.detach(), dim=1).numpy()
    # print("The predicted label is: {} ({})".format(np.argmax(prediction), prediction))
    # print()
    # print()
    #
    # current_token = val_input_ids[sentence_index][token_index]
    # print("Attentions for token: {} ".format(tokenizer.decode(current_token)))
    # print()
    # #
    # for index, token in enumerate(val_input_ids[sentence_index]):
    #     decoded_token = tokenizer.decode(token)
    #     print("{}({:.2f})".format(decoded_token, last_layer_avg[token_index][index].item()), end=' ')
    #
    # print()
    # print()
    #
    # average_over_layers = torch.mean(last_layer_avg, dim=0, keepdim=False)
    #
    # for index, token in enumerate(val_input_ids[sentence_index]):
    #     decoded_token = tokenizer.decode(token)
    #     print("{}({:.2f})".format(decoded_token, average_over_layers[index].item()), end=' ')
    #
    # print()
    print()

    # matplotlib.heatmap


if __name__ == '__main__':
    main()

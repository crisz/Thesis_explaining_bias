import numpy as np
import torch

from dataset_utils.load_misogyny import load_misogyny_val_dataset
from dataset_utils.load_sst2 import load_sst2_val_dataset
from explainers.ShapWrapper import ShapWrapper
from utils.cuda import get_device
from utils.huggingface import get_tokens_from_sentences
from utils.save_model import load_model
import shap


def main():
    model, tokenizer = load_model('bert-base-uncased-fine-tuned-misogyny')
    val_labels, val_sentences = load_misogyny_val_dataset()
    val_input_ids, val_attention_masks, _ = get_tokens_from_sentences(val_sentences, tokenizer=tokenizer)

    print("*", val_sentences.shape)
    device = get_device()

    model.to(device)
    model.eval()

    sentence_index = 3
    token_index = 8

    out = model.forward(
        val_input_ids[0:20].to(device),
        val_attention_masks[0:20].to(device),
        # val_input_ids.to(device),
        # val_attention_masks.to(device),
        token_type_ids=None,
        labels=None,
        return_dict=True,
        output_attentions=True
    )

    # TODO: vedi note. Fare media tra tutte le teste
    current_sentence = val_sentences[sentence_index]

    # Get the last layer, the final one
    last_layer_attention = out.attentions[3]  # shape: [sentences_number, heads, length, length]

    # We only have one sentece, let's examine
    last_layer_current_sentence = last_layer_attention[0]

    # Perform average over heads
    last_layer_avg = torch.mean(last_layer_current_sentence, dim=0, keepdim=False)  # shape: LxL = 64x64
    # The above is what "BERT dark secrets" paper refers as "self-attention maps"
    #
    for i, hidden_state in enumerate(out.hidden_states):
        print(f"hs-{i}", hidden_state.shape)

    # for input_id in val_input_ids:
    #     print("iid", input_id.shape)

    print("Examining the following sentence: ")
    print(current_sentence)
    print()
    print("The ground truth label is: {}".format(val_labels[sentence_index][0]))
    print("The predicted label is: {}".format(np.argmax(out.logits.detach().numpy())))
    print()
    print()

    wrapped = ShapWrapper(model=model)
    # Provare kernel explainer (HCL) https://aclanthology.org/2021.hackashop-1.3.pdf
    embedding = out.embedding_outputs
    e = shap.DeepExplainer(wrapped, embedding[(sentence_index+1):(sentence_index+300)])
    out, emb = e.explainer.model.forward(embedding[(sentence_index+1):(sentence_index+300)], embedding=True)
    print("***", emb.shape)
    data = emb[sentence_index:sentence_index+1]
    shap_values = e.shap_values(data, ranked_outputs=10)[0]

    print("shap_values output shape: ", shap_values.shape)

    s2 = torch.Tensor(shap_values[0])
    s2 = torch.sum(s2, dim=1)
    # s2 = torch.nn.functional.softmax(s2)

    print(s2)

    print(val_attention_masks[sentence_index])

    for index, token in enumerate(val_input_ids[sentence_index]):
        decoded_token = tokenizer.decode(token)
        print("{}({:.2f})".format(decoded_token, s2[index].item()), end=' ')


if __name__ == '__main__':
    main()

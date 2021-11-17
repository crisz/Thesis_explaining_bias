import torch
import numpy as np

from utils.huggingface import get_tokens_from_sentences


class ShapWrapper(torch.nn.Module):
    def __init__(self, model, tokenizer):
        super(ShapWrapper, self).__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, data, embedding=False):
        print("%%%% Received data with shape ", data.shape)
        # if len(data.shape) == 2:
        #     data = data.reshape(-1, 64, 768)

        val_input_ids, val_attention_masks, _ = get_tokens_from_sentences(data.reshape(-1), tokenizer=self.tokenizer)
        print("val_input_ids", val_input_ids.shape)
        outputs = self.model(
            input_ids=val_input_ids,
            attention_mask=val_attention_masks,
            token_type_ids=None,
            labels=None,
            return_dict=True,
            output_attentions=True
        )

        predictions = torch.nn.functional.softmax(outputs.logits, dim=1)

        if not embedding:
            return predictions.detach().numpy()
        else:
            return predictions, outputs.embedding_outputs

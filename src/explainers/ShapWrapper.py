import torch
import numpy as np

from utils.cuda import get_device
from utils.huggingface import get_tokens_from_sentences
from sklearn.preprocessing import OrdinalEncoder


class ShapWrapper(torch.nn.Module):
    def __init__(self, model, tokenizer, encoder):
        super(ShapWrapper, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = get_device()
        self.encoder = encoder

    def forward(self, data, embedding=False):
        print("%%%% Received data with shape ", data.shape)
        # if len(data.shape) == 2:
        #     data = data.reshape(-1, 64, 768)
        data = self.encoder.inverse_transform(data.reshape(-1, 64))
        data = np.array([' '.join(sentence) for sentence in data])

        val_input_ids, val_attention_masks, _ = get_tokens_from_sentences(data, tokenizer=self.tokenizer)
        print("val_input_ids", val_input_ids.shape)
        val_input_ids = val_input_ids.to(self.device)
        val_attention_masks = val_attention_masks.to(self.device)
        self.model.to(self.device)

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
            return predictions.detach().to('cpu').numpy()
        else:
            return predictions, outputs.embedding_outputs

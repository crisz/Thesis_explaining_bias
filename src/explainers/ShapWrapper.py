import torch
import numpy as np
from tqdm import tqdm

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

        ids_length = len(val_input_ids)
        batch_size = 2
        iterations = ids_length//batch_size + 1

        total_prediction = torch.empty((ids_length, 2))
        for i in tqdm(range(iterations)):
            offset_0 = i * batch_size
            offset_1 = i * batch_size + 1
            batch_ids = val_input_ids[offset_0:offset_1]
            mask_ids = val_attention_masks[offset_0:offset_1]
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch_ids,
                    attention_mask=mask_ids,
                    token_type_ids=None,
                    labels=None,
                    return_dict=True,
                    output_attentions=True
                )
            total_prediction[offset_0:offset_1] = torch.nn.functional.softmax(outputs.logits, dim=1)

        if not embedding:
            return total_prediction.detach().to('cpu').numpy()
        else:
            return total_prediction, outputs.embedding_outputs

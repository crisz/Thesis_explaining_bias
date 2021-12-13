from abc import ABC

import torch
import numpy as np
from tqdm import tqdm

from utils.cuda import get_device
from utils.huggingface import get_tokens_from_sentences
from sklearn.preprocessing import OrdinalEncoder


class DeepShapWrapper(torch.nn.Module, ABC):
    def __init__(self, model):
        super(DeepShapWrapper, self).__init__()
        self.model = model
        self.device = get_device()

    def forward(self, data):
        self.model.to(self.device)

        data_len = len(data)
        batch_size = 32
        iterations = data_len//batch_size + 1

        total_prediction = torch.empty((data_len, 2))
        for i in tqdm(range(iterations)):
            offset_0 = i * batch_size
            offset_1 = (i + 1) * batch_size

            batch_ids = data[offset_0:offset_1]
            real_size = len(batch_ids)

            if real_size == 0:
                continue

            mask_ids = torch.from_numpy(np.array((0,)*64*real_size).reshape(real_size, -1))
            self.model.eval()
            outputs = self.model(
                inputs_embeds=batch_ids,
                attention_mask=mask_ids,
                token_type_ids=None,
                labels=None,
                return_dict=True,
                output_attentions=True
            )
            total_prediction[offset_0:offset_1] = outputs.logits

        total_prediction = torch.softmax(total_prediction, dim=1)
        print(total_prediction)
        return total_prediction

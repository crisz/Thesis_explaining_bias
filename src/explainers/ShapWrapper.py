import torch
import numpy as np


class ShapWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ShapWrapper, self).__init__()
        self.model = model

    def forward(self, data, embedding=False):
        outputs = self.model(
            inputs_embeds=data,
            attention_mask=torch.Tensor(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).repeat(data.shape[0], axis=0).reshape((-1, 64))),
            token_type_ids=None,
            labels=None,
            return_dict=True,
            output_attentions=True

        )
        predictions = torch.nn.functional.softmax(outputs.logits, dim=1)

        if not embedding:
            return predictions
        else:
            return predictions, outputs.embedding_outputs

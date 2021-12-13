from pathlib import Path

import numpy as np
import torch
from torch.utils.data import SequentialSampler, DataLoader, TensorDataset
from tqdm import tqdm

from dataset_utils.load_misogyny import load_misogyny_val_dataset
from utils.cuda import get_device
from utils.huggingface import get_tokens_from_sentences
from utils.save_model import load_model
import argparse


def main(args):
    model, tokenizer = load_model('bert-base-uncased-fine-tuned-misogyny')
    val_labels, val_sentences = load_misogyny_val_dataset()

    # val_labels = val_labels[30:31]
    # val_sentences = val_sentences[30:31]
    val_labels = torch.Tensor(val_labels).long()
    val_input_ids, val_attention_masks, _ = get_tokens_from_sentences(val_sentences, tokenizer=tokenizer)
    device = get_device()

    batch_size = 32

    val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)

    val_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size
    )

    model.to(device)
    model.eval()
    model.zero_grad()
    out_file = Path(args.output_file)

    if not out_file.exists():
        print(f'File at path {out_file} does not exist. Creating...')
    else:
        print(f'File at path {out_file} does exist. Will append')

    with open(out_file, "w+") as out:
        headers = ('index', 'source', 'original', 'misclassification_type', 'logits', 'orig_label', 'pred_label')
        out.write('\t'.join(headers)+'\n')
        for i, batch in tqdm(enumerate(val_dataloader)):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                outputs = model.forward(b_input_ids,
                                        token_type_ids=None,
                                        attention_mask=b_input_mask,
                                        labels=b_labels,
                                        return_dict=True)

                for offset, logit in enumerate(outputs.logits):
                    j = i*batch_size+offset
                    real_label = (b_labels[offset] == 1).item()
                    predicted_label = np.argmax(logit.to('cpu').detach().numpy()) == 1

                    misclassification_type = None
                    # Case False Negative
                    if real_label and not predicted_label:
                        misclassification_type = 'FN'
                        print(f"FN ~ index {j} ~ {val_sentences[j]}")

                    # Case False Positive
                    if not real_label and predicted_label:
                        misclassification_type = 'FP'
                        print(f"FP ~ index {j} ~ {val_sentences[j]}")

                    if misclassification_type:
                        orig_label = 1 if misclassification_type == 'FN' else 0
                        pred_label = 1 if orig_label == 0 else 0
                        row = (str(j), 'test.tsv', val_sentences[j], misclassification_type, str(logit), str(orig_label), str(pred_label))
                        out.write('\t'.join(row)+'\n')


if __name__ == '__main__':
    # Parse arguments from command line
    parser = argparse.ArgumentParser()

    parser.add_argument('--output-file',
                        help='Output file to store false positive and false negative',
                        default='./misclassified.out.tsv',
                        action='store_true')

    args = parser.parse_args()
    main(args)

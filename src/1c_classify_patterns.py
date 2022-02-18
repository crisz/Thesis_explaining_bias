import argparse

import numpy as np
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm

from dataset_utils.load_misogyny import load_misogyny_val_dataset
from model.pattern_classifier import PatternClassifier
from utils.cuda import get_device
from utils.huggingface import get_tokens_from_sentences
from utils.images import load_pattern, get_pattern_classifier, increase_contrast
from utils.save_model import load_model
import matplotlib.pyplot as plt


def main(args):
    predictions = []
    model, tokenizer = load_model('bert-base-uncased-fine-tuned-misogyny')
    val_labels, val_sentences = load_misogyny_val_dataset()
    val_input_ids, val_attention_masks, _ = get_tokens_from_sentences(val_sentences, tokenizer=tokenizer)

    device = get_device()

    model.to(device)
    model.eval()
    results = np.zeros((4, 8))
    for epoch, n in enumerate(range(8)):
        print(">> Epoch ", epoch)
        threshold = (10-n)/10 - 0.1
        tp_sum = 0
        fp_sum = 0
        tn_sum = 0
        fn_sum = 0
        tp_count = 0.0001
        fp_count = 0.0001
        tn_count = 0.0001
        fn_count = 0.0001
        for index in tqdm(range(len(val_input_ids[:100]))):
            out = model.forward(
                val_input_ids[index:index+1].to(device),
                val_attention_masks[index:index+1].to(device),
                token_type_ids=None,
                labels=None,
                return_dict=True,
                output_attentions=True
            )
            prediction = dict()
            prediction['index'] = index
            prediction['attention'] = [attentions.detach().numpy() for attentions in out.attentions]
            for i, attention in enumerate(prediction['attention']):
                U, S, V = torch.Tensor.svd(torch.from_numpy(attention), some=False, compute_uv=True)
                bound = torch.finfo(S.dtype).eps * max(U.shape[1], V.shape[1])
                # noinspection PyTypeChecker
                greater_than_bound: torch.Tensor = S > bound
                # noinspection PyArgumentList
                basis_start_index = torch.max(torch.sum(greater_than_bound, dtype=int, axis=2))
                null_space = U[:, :, :, basis_start_index:]
                B = torch.matmul(torch.from_numpy(attention), null_space)
                transpose_B = torch.transpose(B, -1, -2)
                projection_attention = torch.matmul(null_space, transpose_B)
                projection_attention = torch.transpose(projection_attention, -1, -2)
                effective_attention = torch.sub(torch.from_numpy(attention), projection_attention)
                prediction['attention'][i] = effective_attention[0]

            decoded_tokens = [tokenizer.decode(token) for token in val_input_ids[index]]
            try:
                dim = decoded_tokens.index("[ P A D ]")
            except:
                dim = 64
            tokens = np.zeros((dim,))
            pred = torch.argmax(out.logits).detach()

            patterns = np.stack(prediction['attention'], axis=0).reshape((-1, 64, 64))
            patterns = patterns[:, :dim, :dim]
            for i, pattern in enumerate(patterns):
                x = increase_contrast(pattern)
                for index_column in range(dim):
                    if index_column == 0 or index_column == dim-1:
                        continue
                    mat = np.zeros((dim, dim))
                    mat[:, index_column] = 1.
                    result = np.multiply(mat, x).sum()
                    if result/dim > threshold:
                        tokens[index_column] += 1
                    # if index_column == 12:
                    #     print(i//12, i%12)
            # print('\t'.join([str(x) for x in tokens]))
            # sum_tokens = tokens[1:-1].sum()
            sum_tokens = tokens.sum()
            real = val_labels[index][0]
            # print(f"Index: {index}; Predicted: {pred}; Real: {real}; Sum: {sum_tokens}")

            if real == 1 and pred == 1:
                tp_sum += sum_tokens / dim
                tp_count += 1
            if real == 0 and pred == 1:
                fp_sum += sum_tokens / dim
                fp_count += 1
            if real == 1 and pred == 0:
                fn_sum += sum_tokens / dim
                fn_count += 1
            if real == 0 and pred == 0:
                tn_sum += sum_tokens / dim
                tn_count += 1
            # print("TP: {}, FP: {}, FN: {}, TN: {}".format(
            #     tp_sum / tp_count,
            #     fp_sum / fp_count,
            #     fn_sum / fn_count,
            #     tn_sum / tn_count
            # ))
        results[0, epoch] = tp_sum / tp_count
        results[1, epoch] = fp_sum / fp_count
        results[2, epoch] = fn_sum / fn_count
        results[3, epoch] = tn_sum / tn_count
        print(results)
    print(results)
    np.save('./results.npy', results)
    print('>> done')

    coords = np.arange(0.1, 0.9, 0.1)
    t = ['TP', 'FP', 'FN', 'TN']
    for i in range(1, 3):
        plt.plot(coords, results[i], label=t[i])
        plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern-id',
                        help='Pattern id',
                        default=None,
                        type=int)

    parser.add_argument('--method',
                        help='Type of attention',
                        default='normal',
                        type=str)  # TODO restrict choices

    args = parser.parse_args()
    main(args)

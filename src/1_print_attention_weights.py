import argparse
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from config import SAVE_PATTERN_PATH
from dataset_utils.load_misogyny import load_misogyny_val_dataset
from utils.cuda import get_device
from utils.entropy import compute_negative_entropy
from utils.huggingface import get_tokens_from_sentences
from utils.images import save_map
from utils.read_json import load_ids, JSON_FILE
from utils.save_model import load_model
import matplotlib.pyplot as plt


def main(args):
    model, tokenizer = load_model('bert-base-uncased-fine-tuned-misogyny')
    val_labels, val_sentences = load_misogyny_val_dataset()
    val_input_ids, val_attention_masks, _ = get_tokens_from_sentences(val_sentences, tokenizer=tokenizer)

    device = get_device()

    model.to(device)
    model.eval()

    predictions = []

    path = args.json_file if args.json_file is not None else JSON_FILE
    save_path = args.save_path if args.save_path is not None else SAVE_PATTERN_PATH
    folder = args.folder
    method = args.method

    print(f">> Performing predictions of indices found at path {path}")

    for index in load_ids(path):
        print(f">> Predicting index {index}")
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
        prediction['attention'] = [attentions.detach().numpy()[0] for attentions in out.attentions]
        if method == 'effective':
            prediction['value'] = [value.detach().numpy() for value in out.value]
        if method == 'entropy':
            prediction['mask'] = val_attention_masks[index:index+1].to(device)

        predictions.append(prediction)

    print(f">> Prediction done. Saving attention maps at path {save_path}")
    for prediction in tqdm(predictions):
        current_index = prediction['index']
        current_attention = prediction['attention']
        current_value = None
        if method == 'effective':
            current_value = prediction['value']
        current_mask = None
        if method == 'entropy':
            current_mask = prediction['mask']

        current_save_path = save_path / folder / f'index_{current_index}'
        current_save_path_npy = current_save_path / 'npy'
        current_save_path_png = current_save_path / 'png'

        if not current_save_path_npy.exists():
            os.makedirs(current_save_path_npy)

        if not current_save_path_png.exists():
            os.makedirs(current_save_path_png)

        # Save attention map for every couple (layer, head)
        decoded_tokens = [tokenizer.decode(token) for token in val_input_ids[current_index]]
        real_sentence_length = decoded_tokens.index("[ P A D ]")
        decoded_tokens = decoded_tokens[:real_sentence_length]
        x_labels = decoded_tokens
        y_labels = decoded_tokens

        total_output = np.empty((12*real_sentence_length, 12*real_sentence_length))
        total_output_i = 0
        total_output_j = 0

        if method == 'effective':
            effective_attention_map = []
            for current_layer in range(12):
                U, S, V = torch.Tensor.svd(torch.from_numpy(current_value[current_layer]), some=False, compute_uv=True)
                bound = torch.finfo(S.dtype).eps * max(U.shape[1], V.shape[1])
                # noinspection PyTypeChecker
                greater_than_bound: torch.Tensor = S > bound
                # noinspection PyArgumentList
                basis_start_index = torch.max(torch.sum(greater_than_bound, dtype=int, axis=2))
                null_space = U[:, :, :, basis_start_index:]
                B = torch.matmul(torch.from_numpy(current_attention[current_layer]), null_space)
                transpose_B = torch.transpose(B, -1, -2)
                projection_attention = torch.matmul(null_space, transpose_B)
                projection_attention = torch.transpose(projection_attention, -1, -2)
                effective_attention = torch.sub(torch.from_numpy(current_attention[current_layer]), projection_attention)
                effective_attention_map.append(effective_attention[0].numpy())
            current_attention = effective_attention_map
        elif method == 'entropy':
            torch_attention = [torch.from_numpy(np.expand_dims(_attention, axis=0)) for _attention in current_attention]
            _, [current_attention] = compute_negative_entropy(torch_attention, current_mask, True)
            y_labels = list(range(12))

        for current_layer in range(12):
            total_output_j_offset = total_output_j + real_sentence_length
            for current_head in range(12):
                if method == 'entropy':
                    sliced_attention_map = current_attention[current_layer]
                    sliced_attention_map = sliced_attention_map[:real_sentence_length]
                    # Positive entropy
                    sliced_attention_map = -sliced_attention_map
                    # Removing CLS and SEP
                    sliced_attention_map = sliced_attention_map[:, 1:-1]
                    # Rescaling
                    sliced_attention_map = (1 / sliced_attention_map).log()
                else:
                    sliced_attention_map = current_attention[current_layer][current_head]
                    sliced_attention_map = sliced_attention_map[:real_sentence_length, :real_sentence_length]

                save_map(
                    map=sliced_attention_map,
                    x_labels=x_labels,
                    y_labels=y_labels,
                    path=current_save_path_png / f'layer_{current_layer}_head_{current_head}.png',
                    title=f'Attention map for layer {current_layer} and head {current_head}'
                )

                np.save(current_save_path_npy / f'layer_{current_layer}_head_{current_head}', sliced_attention_map)
                total_output_i_offset = total_output_i + real_sentence_length
                total_output[total_output_i:total_output_i_offset, total_output_j:total_output_j_offset] = sliced_attention_map
                total_output_i = total_output_i_offset
            total_output_j = total_output_j_offset
            total_output_i = 0

        print(">> Saving composition of maps")

        save_map(
            map=total_output,
            x_labels=[],
            y_labels=[],
            path=current_save_path_png / 'final_map.png',
            title='Composition of attention maps'
        )

        # Save attention map for every layer averaging across heads
        for current_layer in range(12):
            sliced_attention_map = current_attention[current_layer]
            sliced_attention_map = np.average(sliced_attention_map, axis=0)
            sliced_attention_map = sliced_attention_map[:real_sentence_length, :real_sentence_length]

            save_map(
                map=sliced_attention_map,
                x_labels=x_labels,
                y_labels=y_labels,
                path=current_save_path_png / f'layer_{current_layer}_avg_heads.png',
                title=f'Attention map for layer {current_layer} averaging over heads'
            )

            np.save(current_save_path_npy / f'layer_{current_layer}_avg_heads', sliced_attention_map)

    print(">> Patterns correctly saved.")
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--json-file',
                        help='JSON file that stores the indices of selected false positive and false negative',
                        default=None,
                        type=str)

    parser.add_argument('--save-path',
                        help='Path to save the generated images',
                        default=None,
                        type=str)

    parser.add_argument('--folder',
                        help='Task-specific folder to store the results',
                        required=True,
                        type=str)

    parser.add_argument('--method',
                        help='Type of attention',
                        default='normal',
                        type=str)  # TODO restrict choices

    args = parser.parse_args()
    main(args)

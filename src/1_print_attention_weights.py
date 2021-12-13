import argparse
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import SAVE_PATTERN_PATH
from dataset_utils.load_misogyny import load_misogyny_val_dataset
from utils.cuda import get_device
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
        predictions.append(prediction)

    print(f">> Prediction done. Saving attention maps at path {save_path}")
    for prediction in tqdm(predictions):
        current_index = prediction['index']
        current_attention = prediction['attention']

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

        # total_output = np.stack(current_attention, axis=0).transpose((0, 2, 1, 3)).reshape(12*64, 12*64)
        # print(">> Saving composition of maps")
        # save_map(
        #     map=total_output,
        #     x_labels=[],
        #     y_labels=[],
        #     path=current_save_path_png / 'final_map.png',
        #     title='Composition of attention maps'
        # )
        # exit(-1)

        total_output = np.empty((12*real_sentence_length, 12*real_sentence_length))
        total_output_i = 0
        total_output_j = 0

        for current_layer in range(12):
            total_output_j_offset = total_output_j + real_sentence_length
            for current_head in range(12):
                sliced_attention_map = current_attention[current_layer][current_head]
                sliced_attention_map = sliced_attention_map[:real_sentence_length, :real_sentence_length]

                save_map(
                    map=sliced_attention_map,
                    x_labels=decoded_tokens,
                    y_labels=decoded_tokens,
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
        print(total_output.shape)

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
                x_labels=decoded_tokens,
                y_labels=decoded_tokens,
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

    args = parser.parse_args()
    main(args)

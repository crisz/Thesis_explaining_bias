import argparse

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from dataset_utils.load_immigration import load_immigration_train_dataset, load_immigration_val_dataset
from dataset_utils.load_misogyny import load_misogyny_val_dataset, load_misogyny_train_dataset
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch

from utils.cuda import get_device
from utils.huggingface import get_tokens_from_sentences
from utils.save_model import save_model


def main(args):
    dataset = args.dataset
    train_labels, train_sentences, val_labels, val_sentences = (None, None, None, None)
    if dataset == 'misog':
        train_labels, train_sentences = load_misogyny_train_dataset()
        val_labels, val_sentences = load_misogyny_val_dataset()
    elif dataset == 'mig':
        train_labels, train_sentences = load_immigration_train_dataset()
        val_labels, val_sentences = load_immigration_val_dataset()
    else:
        print(f"The dataset {dataset} is not recognized")
        exit(-1)

    print(train_labels)
    print(train_sentences)
    print(val_labels)
    print(val_sentences)

    train_input_ids, train_attention_masks, train_tokenizer = get_tokens_from_sentences(train_sentences)

    train_labels = torch.Tensor(train_labels).long()
    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)

    val_input_ids, val_attention_masks, _ = get_tokens_from_sentences(val_sentences)

    val_labels = torch.Tensor(val_labels).long()
    val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)

    model = train(train_dataset=train_dataset, val_dataset=val_dataset, args=args)
    save_model(model, train_tokenizer, f'bert-base-uncased-fine-tuned-{dataset}')


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def train(train_dataset, val_dataset, args):
    batch_size = 32

    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size
    )

    val_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size
    )

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        output_attentions=True,
        output_hidden_states=True,
    )

    device = get_device()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.epsilon)

    epochs = args.epochs

    model.train()
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=len(train_dataloader) * epochs)

    for epoch in range(epochs):
        print("Epoch {} out of {}".format(epoch+1, epochs))
        total_train_loss = 0
        total_train_accuracy = 0

        for batch in tqdm(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()
            result = model.forward(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels,
                                   return_dict=True)
            loss = result.loss
            total_train_loss += loss.item()
            loss.backward()

            # Avoid the exploding gradient problem:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            logits = result.logits.detach().to('cpu').numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_train_accuracy += flat_accuracy(logits, label_ids)

            # TODO: Update the learning rate.
            scheduler.step()
        avg_train_loss = total_train_loss / len(train_dataloader)
        print("Average train loss is {}".format(avg_train_loss))
        avg_train_accuracy = total_train_accuracy / len(train_dataloader)
        print(">> Train Accuracy: {0:.2f}".format(avg_train_accuracy))

        print("Calculating the val accuracy...")
        total_eval_loss = 0
        total_eval_accuracy = 0

        model.eval()
        for batch in tqdm(val_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                result = model.forward(b_input_ids,
                                       token_type_ids=None,
                                       attention_mask=b_input_mask,
                                       labels=b_labels,
                                       return_dict=True)

            loss = result.loss
            logits = result.logits

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().to('cpu').numpy()
            print(logits)
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)

            # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
        print(">> Val Accuracy: {0:.2f}".format(avg_val_accuracy))
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        help='Dataset for the training',
                        default=None,
                        type=str)

    parser.add_argument('--lr',
                        help='Learning rate for the training',
                        default=2e-5,
                        type=float)

    parser.add_argument('--epsilon',
                        help='Epsilon for the training',
                        default=1e-9,
                        type=float)

    parser.add_argument('--epochs',
                        help='Epochs for the training',
                        default=4,
                        type=int)

    args = parser.parse_args()
    main(args)

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from tqdm import tqdm

from dataset_utils.load_sst2 import load_train_dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch

from utils.cuda import get_device


def get_tokens_from_sentences(sentences):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # Tokenize all of the sentences and map the tokens to their word IDs.
    input_ids = []
    attention_masks = []

    for sentence in sentences:
        encoded_dict = tokenizer.encode_plus(
            sentence,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            truncation=True,
            max_length=64,  # Pad & truncate all sentences.
            padding='max_length',
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks


def train(train_labels, train_input_ids, train_attention_masks):
    batch_size = 32
    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
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
    # TODO: learning rate and epsilon
    optimizer = AdamW(model.parameters())

    epochs = 4

    model.train()
    for epoch in range(epochs):
        print("Epoch {} out of {}".format(epoch, epochs))
        total_train_loss = 0
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
            logits = result.logits
            total_train_loss += loss.item()
            loss.backward()

            # Avoid the exploding gradient problem:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # TODO: Update the learning rate.
            # scheduler.step()
        avg_train_loss = total_train_loss / len(train_dataloader)
        print("Average train loss is {}".format(avg_train_loss))
        print("Calculating the train accuracy...")


def main():
    train_labels, train_sentences = load_train_dataset()
    train_input_ids, train_attention_masks = get_tokens_from_sentences(train_sentences)
    train_labels = train_labels.astype(np.int)
    train_labels = np.array(train_labels).reshape(-1, 1)
    train_labels = torch.Tensor(train_labels).long()
    print("dtype is ", train_labels.dtype)
    train(train_labels, train_input_ids, train_attention_masks)


if __name__ == '__main__':
    main()

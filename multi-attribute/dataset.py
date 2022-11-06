import os

import re
from io import open # pylint: disable=redefined-builtin

import json
from abc import *

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from calculate_diversity import preprocess_text

import numpy as np

def tokenize(tokenizer, raw_text):
    max_len = tokenizer.max_len

    if len(raw_text) > max_len:
        raw_text = raw_text[:max_len]

    tokens = tokenizer.encode(raw_text, add_special_tokens=True)
    tokens = torch.tensor(tokens).long()

    tokenizer.pad_token = '[PAD]'
    if tokens.size(0) < max_len:
        padding = torch.zeros(max_len - tokens.size(0)).long()
        padding.fill_(tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
        tokens = torch.cat([tokens, padding])

    return tokens


def create_tensor_dataset(inputs, labels):
    assert len(inputs) == len(labels)

    inputs = torch.stack(inputs)  # (N, T)
    labels = torch.stack(labels).unsqueeze(1)  # (N, 1)

    dataset = TensorDataset(inputs, labels)

    return dataset

def create_dataset(source_file, tokenizer, label=0):
    source_file = open(source_file, "r")
    examples = preprocess_text(source_file)
    # print(examples[:10])

    all_tokens = [tokenize(tokenizer, example) for example in examples]
    labels = torch.tensor(label).long()
    all_labels = []
    for _ in range(len(all_tokens)):
        all_labels.append(labels)
    
    source_dataset = create_tensor_dataset(all_tokens, all_labels)

    return source_dataset
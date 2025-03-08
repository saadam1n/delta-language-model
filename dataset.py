import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import csv
import tqdm
import re
import sys
import os

import tiktoken

class TextDataset(torch.utils.data.Dataset):
    """Tool to read data from the Medium dataset."""
    def __init__(self, tokenizer: tiktoken.Encoding, min_length: int, max_length: int, path: str):
        super(TextDataset, self).__init__()

        self.tokenizer = tokenizer
        self.min_length = min_length
        self.max_length = max_length
        self.path = path

        self.samples = []

        cache_path = path + ".tokenized_cache"
        if not os.path.exists(cache_path):
            with open(self.path, "r") as f:
                reader = csv.reader(f)

                # skip the header row
                next(reader, None)

                for row in tqdm.tqdm(iterable=reader, desc="Reading and tokenizing dataset..."):
                    article = row[1]

                    text = article.strip()
                    if not text:
                        continue

                    if not text.endswith("."):
                        text += "." 

                    tokenization = self.tokenizer.encode(article, disallowed_special=(self.tokenizer.special_tokens_set - {'<|endoftext|>'}))

                    # add end of sequence token
                    tokenization.append(self.tokenizer.max_token_value)

                    if len(tokenization) > self.max_length or len(tokenization) < self.min_length:
                        continue

                    tokenization = torch.tensor(tokenization, dtype=torch.long)

                    self.samples.append(tokenization)

            with open(cache_path, "w") as f:
                for tokenization in tqdm.tqdm(iterable=self.samples, desc="Building cache..."):
                    for token in tokenization:
                        f.write(f"{token} ")
                    f.write("\n")
        else:
            with open(cache_path, "r") as f:
                lines = f.readlines()

                self.samples = [
                    torch.tensor(
                        [
                            int(token) for token in line.split()
                        ], dtype=torch.long
                    )
                    for line in tqdm.tqdm(iterable=lines, desc="Reading dataset from cache...")
                ]

        # zero pad everything
        self.truncated_lengths = []
        for i in range(len(self.samples)):
            truncated_length = self.samples[i].shape[0]

            self.truncated_lengths.append(truncated_length)

            padding_size = self.max_length - truncated_length
            self.samples[i] = torch.cat((self.samples[i], torch.ones(padding_size, dtype=torch.long) * tokenizer.max_token_value), dim=0)

        print(f"Dataset had {len(self)} items that satisfied filtration requirements.")

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)
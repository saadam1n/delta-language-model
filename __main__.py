from model import LanguageModel
from dataset import TextDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tiktoken
import time
import torchshow as ts

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        raise RuntimeError("Unable to use CUDA!")

    num_transformer_layers=16
    num_attention_heads=16
    min_length = 64
    context_length=384
    embedding_dim=128
    head_embedding_dim=8
    tokenizer = tiktoken.get_encoding("r50k_base")
    num_epochs = 16
    batch_size = 32

    dataloader = torch.utils.data.DataLoader(
        TextDataset(
            tokenizer=tokenizer,
            min_length=min_length,
            max_length=context_length,
            path="data/medium_articles.csv"
        ),
        batch_size=batch_size,
        shuffle=True
    )

    delta = LanguageModel(num_transformer_layers=num_transformer_layers, num_attention_heads=num_attention_heads, context_length=context_length, embedding_dim=embedding_dim, head_embedding_dim=8, tokenizer=tokenizer)
    delta = delta.to(device)

    num_parameters = sum(param.numel() for param in delta.parameters())
    print(f"Training LLM with {num_parameters} parameters.")

    optimizer = torch.optim.Adam(delta.parameters(), lr=0.001)
    scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.85)

    loss_history = []

    delta.train()
    for epoch_idx in range(num_epochs):
        print(f"Processing epoch {epoch_idx}")

        total_loss = 0.0
        num_sequences = 0

        for batch_idx, data in enumerate(dataloader):
            #print(f"\tProcessing training batch {batch_idx}")

            data = data.to(device)
           
            optimizer.zero_grad()

            logits = delta.forward_tokens(data)

            N, L, _ = logits.shape

            # for any given token, we want to predict the next token
            # for this reason we shift data forward by 1 and pad with -1
            shifted_tokens = torch.cat((data[:, 1:], -torch.ones_like(data[:, :1])), dim=1)

            flattened_logits = logits.view(N * L, -1)
            flattened_tokens = shifted_tokens.flatten()

            loss = F.cross_entropy(input=flattened_logits, target=flattened_tokens, ignore_index=-1)

            flattened_logits.retain_grad()
            loss.backward()

            optimizer.step()

            total_loss += loss.item() * data.shape[0]
            num_sequences += data.shape[0]

            print(f"\tLoss in batch {batch_idx}\twas {loss.item()}")

        average_loss = total_loss / num_sequences
        loss_history.append(average_loss)

        print("\nLoss history so far:")
        for i, loss in enumerate(loss_history):
            print(f"\tLoss in epoch {i}\t: {loss}")

        scheduler.step()

        torch.save(delta.state_dict(), f"data/model-{epoch_idx}.pt")

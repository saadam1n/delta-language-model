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

    num_transformer_layers=8
    num_attention_heads=32
    min_length = 128
    context_length=511
    embedding_dim=384
    head_embedding_dim=16
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

    torch.set_float32_matmul_precision("high")

    delta = LanguageModel(num_transformer_layers=num_transformer_layers, num_attention_heads=num_attention_heads, context_length=context_length, embedding_dim=embedding_dim, head_embedding_dim=8, tokenizer=tokenizer)
    delta = delta.to(device)

    num_parameters = sum(param.numel() for param in delta.parameters()) / 1000000
    print(f"Training LLM with {num_parameters}M parameters.")

    optimizer = torch.optim.Adam(delta.parameters(), lr=0.001)
    scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.85)

    loss_history = []

    delta.train()
    for epoch_idx in range(num_epochs):
        print(f"Processing epoch {epoch_idx}")

        total_loss = 0.0
        num_sequences = 0

        epoch_start_time = time.time()
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)
           
            optimizer.zero_grad()

            loss = delta(data)
            loss.backward()

            optimizer.step()

            total_loss += loss.item() * data.shape[0]
            num_sequences += data.shape[0]

            print(f"\tPerpelexity in batch {batch_idx}\twas {loss.exp().item()}")
        duration = time.time() - epoch_start_time

        print(f"Epoch took {duration} seconds to complete; ETA for rest of training is {(num_epochs - epoch_idx - 1) * duration}")

        average_loss = total_loss / num_sequences
        loss_history.append(average_loss)

        print("\nLoss history so far:")
        for i, loss in enumerate(loss_history):
            print(f"\tLoss in epoch {i}\t: {loss}")

        scheduler.step()

        torch.save(delta.state_dict(), f"data/model-{epoch_idx}.pt")

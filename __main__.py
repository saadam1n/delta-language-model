from model import LanguageModel
from dataset import TextDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tiktoken

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        raise RuntimeError("Unable to use CUDA!")

    num_transformer_layers=8
    num_attention_heads=8
    min_length = 64
    context_length=1024
    embedding_dim=64
    head_embedding_dim=8
    tokenizer = tiktoken.get_encoding("r50k_base")
    num_epochs = 128
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
    delta = delta.to(device).half()

    optimizer = torch.optim.Adam(delta.parameters(), lr=0.005)
    scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.85)
    loss_fn = nn.NLLLoss()

    delta.train()
    for epoch_idx in range(num_epochs):
        print(f"Processing epoch {epoch_idx}")

        for batch_idx, data in enumerate(dataloader):
            print(f"\tProcessing training batch {batch_idx}")

            data = data.to(device)

            optimizer.zero_grad()

            logits = delta.forward_tokens(data).transpose(1, 2)

            loss = loss_fn(logits, data)
            loss.backward()

            optimizer.step()

    scheduler.step()


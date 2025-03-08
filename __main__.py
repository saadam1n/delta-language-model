from model import LanguageModel
from dataset import TextDataset

import torch
import tiktoken

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        raise RuntimeError("Unable to use CUDA!")

    tokenizer = tiktoken.get_encoding("r50k_base")

    context_length = 1024

    dataloader = torch.utils.data.DataLoader(
        TextDataset(
            tokenizer=tokenizer,
            min_length=64,
            max_length=context_length,
            path="data/medium_articles.csv"
        ),
        batch_size=32,
        shuffle=True
    )

    delta = LanguageModel(num_transformer_layers=8, num_attention_heads=8, context_length=context_length, embedding_dim=64, head_embedding_dim=8, tokenizer=tokenizer)
    delta = delta.to(device)


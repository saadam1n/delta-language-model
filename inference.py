import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import LanguageModel

import tiktoken

num_transformer_layers=16
num_attention_heads=16
min_length = 64
context_length=384
embedding_dim=128
head_embedding_dim=8
tokenizer = tiktoken.get_encoding("r50k_base")
num_epochs = 16
batch_size = 32

delta = LanguageModel(num_transformer_layers=num_transformer_layers, num_attention_heads=num_attention_heads, context_length=context_length, embedding_dim=embedding_dim, head_embedding_dim=8, tokenizer=tokenizer).to("cuda")
delta.load_state_dict(torch.load("data/model-15-8x8.pt"))
delta.eval()

while True:
    print("Enter your prompt: ", end="")
    prompt = input()


    tokenizer = tiktoken.get_encoding("r50k_base")

    prompt_tokenized = tokenizer.encode(prompt)

    prompt_vector = torch.tensor(
        prompt_tokenized + [tokenizer.max_token_value] * (384 - len(prompt_tokenized)),
        dtype=torch.long,
        device="cuda"
    ).unsqueeze(0)

    print(f"Completion: {prompt}", end="", flush=True)
    for i in range(len(prompt_tokenized), 384):
        # (1, L, P) -> (P)
        next_token_dist = delta.forward_tokens(prompt_vector)[0, i - 1, :].exp()

        if True:
            selected_token_index = torch.searchsorted(next_token_dist.cumsum(dim=0), torch.rand(1, device="cuda")).clamp(min=0, max=tokenizer.max_token_value)
        else:
            selected_token_index = torch.argmax(next_token_dist, dim=0)

        prompt_vector[0, i] = selected_token_index

        print(tokenizer.decode([selected_token_index]), end="", flush=True)

        if selected_token_index == tokenizer.max_token_value:
            print(f"\nEnded sequence with probability {next_token_dist[selected_token_index].item()}")
            break

    print()

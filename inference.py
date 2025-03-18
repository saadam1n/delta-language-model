import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import LanguageModel

import tiktoken

num_transformer_layers=8
num_attention_heads=32
min_length = 128
context_length=511
embedding_dim=384
head_embedding_dim=16
tokenizer = tiktoken.get_encoding("r50k_base")
num_epochs = 16
batch_size = 32

delta = LanguageModel(num_transformer_layers=num_transformer_layers, num_attention_heads=num_attention_heads, context_length=context_length, embedding_dim=embedding_dim, head_embedding_dim=8, tokenizer=tokenizer).to("cuda")
delta.load_state_dict(torch.load("data/model-15-8x32.pt"))
delta.eval()
delta.set_temperature(0.7)

with torch.no_grad():
    while True:
        print("Enter your prompt (enter exit to exit): ", end="")
        prompt = input()

        if prompt == "exit":
            break

        tokenizer = tiktoken.get_encoding("r50k_base")

        prompt_tokenized = tokenizer.encode(prompt)

        prompt_vector = torch.tensor(
            prompt_tokenized + [tokenizer.max_token_value] * (context_length - len(prompt_tokenized)),
            dtype=torch.long,
            device="cuda"
        ).unsqueeze(0)

        print(f"Completion: {prompt}", end="", flush=True)
        tokenization_buf = []
        for i in range(len(prompt_tokenized), context_length):
            # (1, L, P) -> (P)
            next_token_dist = delta(prompt_vector)[0, i, :]

            if True:
                selected_token_index = torch.searchsorted(next_token_dist.cumsum(dim=0), torch.rand(1, device="cuda")).clamp(min=0, max=tokenizer.max_token_value)
            else:
                selected_token_index = torch.argmax(next_token_dist, dim=0)

            if selected_token_index == tokenizer.max_token_value:
                print(f"\n<EOS p={next_token_dist[selected_token_index].item()}>")
                break

            prompt_vector[0, i] = selected_token_index

            tokenization_buf.append(selected_token_index)
            try:
                next_string = tokenizer.decode(tokenization_buf, errors="strict")

                # if we made it this far, we succeeded in decoding and can now clear the buffer
                tokenization_buf.clear()
                print(next_string, end="", flush=True)
            except:
                pass
                
        if len(tokenization_buf) > 0:
            print(tokenizer.decode(tokenization_buf))



        print()

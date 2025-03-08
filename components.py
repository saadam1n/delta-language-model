import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math

class TokenMLP(nn.Module):
    """
    A double layer perceptron network with a skip connection. Automatically performs layer norm on the input.
    """
    def __init__(self, embedding_dim):
        super(TokenMLP, self).__init__()

        self.embedding_dim = embedding_dim

        self.ffn = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.embedding_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(self.embedding_dim * 2),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        )

    def forward(self, x):
        return self.ffn(x) + x

class MultiHeadCasualAttention(nn.Module):
    def __init__(self, embedding_dim, num_attention_heads, head_embedding_dim):
        super(MultiHeadCasualAttention, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.head_embedding_dim = head_embedding_dim

        self.total_head_dim = num_attention_heads * head_embedding_dim

        # linear layers to project our tokens to Q, K, and V
        self.q_linear = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.total_head_dim)
        )

        self.k_linear = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.total_head_dim)
        )

        self.v_linear = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.total_head_dim)
        )

        self.attention_linear = nn.Sequential(
            nn.LayerNorm(self.total_head_dim),
            nn.Linear(self.total_head_dim, self.embedding_dim)
        )

    """
    Returns (transformed tokens, K, V). Does not maintain a KV cache.
    """
    def forward(self, tokens):
        # nn.Linear gives us (N, L, D_embedding_dim) -> (N, L, D_total_head_dim)
        # we want to reformat it as (N, num_attention_heads, L, D_head_embedding_dim)

        N, L, _ = tokens.shape

        q = self.q_linear(tokens).view(N, L, self.num_attention_heads, self.head_embedding_dim).permute((0, 2, 1, 3))
        k = self.v_linear(tokens).view(N, L, self.num_attention_heads, self.head_embedding_dim).permute((0, 2, 3, 1)) # swap 3 and 1 for attention
        v = self.k_linear(tokens).view(N, L, self.num_attention_heads, self.head_embedding_dim).permute((0, 2, 1, 3))

        qkT = torch.matmul(q, k)
        masked_qkT = qkT + torch.triu(torch.ones_like(qkT) * -9999.0, diagonal=1)

        attention_scores = F.softmax(masked_qkT / math.sqrt(self.head_embedding_dim), dim=3)
        weighted_values = torch.matmul(attention_scores, v).permute((0, 2, 1, 3)).reshape(N, L, self.total_head_dim)

        attention_output = self.attention_linear(weighted_values) + tokens

        return (attention_output, k, v)
    
class Transformer(nn.Module):
    def __init__(self, embedding_dim, num_attention_heads, head_embedding_dim):
        super(Transformer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.head_embedding_dim = head_embedding_dim

        self.attention = MultiHeadCasualAttention(
            embedding_dim=self.embedding_dim, 
            num_attention_heads=self.num_attention_heads, 
            head_embedding_dim=self.head_embedding_dim
        )

        self.token_mlp = TokenMLP(embedding_dim=self.embedding_dim)

    def forward(self, tokens):
        attention_output, _, _ = self.attention(tokens)

        token_mlp_output = self.token_mlp(attention_output)

        return token_mlp_output


        

        
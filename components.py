import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

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
            nn.Linear(self.embedding_dim, self.embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(self.embedding_dim * 4, self.embedding_dim)
        )

    def forward(self, x):
        return self.ffn(x) + x

class MultiHeadCasualAttention(nn.Module):
    def __init__(self, embedding_dim, num_attention_heads, head_embedding_dim, manual_attention=False):
        super(MultiHeadCasualAttention, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.head_embedding_dim = head_embedding_dim

        self.total_head_dim = num_attention_heads * head_embedding_dim

        # pre-normalization
        self.pre_norm = nn.LayerNorm(self.embedding_dim)

        # linear layers to project our tokens to Q, K, and V
        self.q_linear = nn.Linear(self.embedding_dim, self.total_head_dim, bias=False)
        self.k_linear = nn.Linear(self.embedding_dim, self.total_head_dim, bias=False)
        self.v_linear = nn.Linear(self.embedding_dim, self.total_head_dim, bias=False)
        self.attention_linear = nn.Linear(self.total_head_dim, self.embedding_dim, bias=False)

        self.manual_attention = manual_attention

    """
    Returns (transformed tokens, K, V). Does not maintain a KV cache.
    """
    def forward(self, tokens):
        # nn.Linear gives us (N, L, D_embedding_dim) -> (N, L, D_total_head_dim)
        # we want to reformat it as (N, num_attention_heads, L, D_head_embedding_dim)

        N, L, _ = tokens.shape

        ln_tokens = self.pre_norm(tokens)

        q = self.q_linear(ln_tokens).view(N, L, self.num_attention_heads, self.head_embedding_dim).permute((0, 2, 1, 3))
        k = self.v_linear(ln_tokens).view(N, L, self.num_attention_heads, self.head_embedding_dim).permute((0, 2, 1, 3))
        v = self.k_linear(ln_tokens).view(N, L, self.num_attention_heads, self.head_embedding_dim).permute((0, 2, 1, 3))

        if self.manual_attention:
            qkT = torch.matmul(q, k.transpose(2, 3))
            masked_qkT = qkT + torch.triu(torch.ones_like(qkT) * -999.0, diagonal=1)

            attention_scores = F.softmax(masked_qkT / math.sqrt(self.head_embedding_dim), dim=3)
            weighted_values = torch.matmul(attention_scores, v)
        else:
            weighted_values = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        aggregated_tokens = weighted_values.permute((0, 2, 1, 3)).reshape(N, L, self.total_head_dim)
        attention_output = self.attention_linear(aggregated_tokens) + tokens

        return (attention_output, k, v)
    
class Transformer(nn.Module):
    def __init__(self, embedding_dim, num_attention_heads, head_embedding_dim, aggresive_checkpointing=False):
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

        self.aggresive_checkpointing = aggresive_checkpointing

    def forward(self, tokens):
        attention_output, _, _ = checkpoint.checkpoint(self.attention, tokens, use_reentrant=False) if self.aggresive_checkpointing else self.attention(tokens)

        token_mlp_output = checkpoint.checkpoint(self.token_mlp, attention_output, use_reentrant=False) if self.aggresive_checkpointing else self.token_mlp(attention_output)

        return token_mlp_output


        

        
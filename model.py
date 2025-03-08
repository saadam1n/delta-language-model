import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from components import Transformer

class LanguageModel(nn.Module):
    """
    The Delta language model.
    """
    def __init__(self, num_transformer_layers, num_attention_heads, context_length, embedding_dim, head_embedding_dim, tokenizer):
        super(LanguageModel, self).__init__()

        self.num_transformer_layers = num_transformer_layers
        self.num_attention_heads = num_attention_heads
        self.context_length = context_length
        self.embedding_dim = embedding_dim
        self.head_embedding_dim = head_embedding_dim
        self.tokenizer = tokenizer
        self.num_embeddings = self.tokenizer.max_token_value + 1

        self.embedding_matrix = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim)
        self.positional_encoding = nn.Embedding(num_embeddings=self.context_length, embedding_dim=self.embedding_dim)

        self.transformer_layers = nn.Sequential(
            *[
                Transformer(embedding_dim=self.embedding_dim, num_attention_heads=self.num_attention_heads, head_embedding_dim=self.head_embedding_dim)
                for _ in range(self.num_transformer_layers)
            ]
        )

    """
    Takes in one-hot vectors. Produces a probability distribution of output tokens.
    """
    def forward_embeddings(self, raw_token_embeddings):
        N, _, _ = raw_token_embeddings.shape

        batched_positional_encodings = self.positional_encoding(
            torch.arange(0, self.context_length, device=raw_token_embeddings.device).view(1, -1).expand(N, -1)
        )

        token_embeddings = raw_token_embeddings + batched_positional_encodings

        latent_next_tokens = checkpoint.checkpoint_sequential(self.transformer_layers, segments=self.num_transformer_layers // 2, input=token_embeddings, use_reentrant=False)

        next_token_probabilities = checkpoint.checkpoint(self.compute_next_token_probabilities, latent_next_tokens, use_reentrant=False)

        return next_token_probabilities
    
    """
    Takes in a list of tokens. Produces a probability distribution of output tokens.
    """
    def forward_tokens(self, token_ids):
        raw_token_embeddings = self.embedding_matrix(token_ids)

        return self.forward_embeddings(raw_token_embeddings)
    
    def compute_next_token_probabilities(self, latent_next_tokens):
        return F.log_softmax(
            torch.matmul(latent_next_tokens, self.embedding_matrix.weight.transpose(0, 1)), dim=2
        )
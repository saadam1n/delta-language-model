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

        self.seq_start_token = nn.Embedding(num_embeddings=1, embedding_dim=self.embedding_dim)
        self.decode_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.temperature = 1.0

    """
    Takes in Token IDs. If in evaluation mode, returns softmax probabilities of the next token. Otherwise, returns the loss.
    """
    def forward(self, token_ids):
        N, _ = token_ids.shape

        token_embeddings = checkpoint.checkpoint(self.fetch_embeddings, token_ids, use_reentrant=False)
        latent_next_tokens = checkpoint.checkpoint_sequential(self.transformer_layers, segments=self.num_transformer_layers // 2, input=token_embeddings, use_reentrant=False)
        
        model_output = None
        if self.training:
            model_output = checkpoint.checkpoint(self.calc_next_token_loss, latent_next_tokens, token_ids, use_reentrant=False)
        else:
            model_output = checkpoint.checkpoint(self.calc_next_token_dist, latent_next_tokens, use_reentrant=False)

        return model_output
    
    """
    Takes in a list of tokens. Returns the embeddings.
    """
    def fetch_embeddings(self, token_ids):
        # we want to replace all padding indices
        cleaned_token_ids = torch.where(token_ids != -1, token_ids, self.tokenizer.max_token_value)
        text_embedding = self.embedding_matrix(cleaned_token_ids) + self.positional_encoding.weight
        start_embedding = self.seq_start_token.weight.view(1, 1, -1).expand(text_embedding.shape[0], -1, -1)

        return torch.cat((start_embedding, text_embedding), dim=1)
    
    def calc_next_token_logits(self, latent_next_tokens):
        return torch.matmul(self.decode_layer_norm(latent_next_tokens), self.embedding_matrix.weight.transpose(0, 1)) / self.temperature
    
    def calc_next_token_dist(self, latent_next_tokens):
        return F.softmax(self.calc_next_token_logits(latent_next_tokens), dim=2)

    def calc_next_token_loss(self, latent_next_tokens, token_ids):
        # the last token doesn't have a next token to predict, so we can slice that out
        latent_next_tokens = latent_next_tokens[:, :-1, :]
        
        N, L, _ = latent_next_tokens.shape

        flattened_logits = self.calc_next_token_logits(latent_next_tokens).view(N * L, -1)
    
        # we do not have to shift the sequence here do tho the sequence start token
        # "<S>  I       like   to   go   to      school <E>" (what the LLM sees as its input) ->
        # "I    like    to     go   to   school  <E>" (what it outputs after slicing out last token predictions)
        # thus we can directly take in our input sequence and use it for cross-entropy loss
        flattened_tokens = token_ids.flatten()

        return F.cross_entropy(input=flattened_logits, target=flattened_tokens, ignore_index=-1)
    
    def set_temperature(self, temp):
        self.temperature = temp
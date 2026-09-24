import numpy as np
import torch
import torch.nn as nn
from utils import SEED, VOCAB_SIZE, D_MODEL, N_HEADS, D_HEAD, N_LAYERS

torch.manual_seed(SEED)

def positional_encoding(seq_length: int, d_model: int) -> torch.Tensor:
    """
    Generate positional encoding for the input sequence.

    Args:
        seq_length (int): Length of the input sequence.
        d_model (int): Dimensionality of the model.

    Returns:
        torch.Tensor: Positional encoding tensor of shape (seq_length, d_model).
    """
    position = torch.arange(0, seq_length).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    pe = torch.zeros(seq_length, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class AttentionModel(nn.Module):
    def __init__(self) -> None:
        super(AttentionModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=D_MODEL)
        self.W_Q = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL), nn.Linear(D_MODEL, D_MODEL)]) # D_MODEL = N_HEADS * D_HEAD
        self.W_K = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL), nn.Linear(D_MODEL, D_MODEL)])
        self.W_V = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL), nn.Linear(D_MODEL, D_MODEL)])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, ablate_heads: list[tuple] = None) -> dict:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length).

        Returns:
            torch.Tensor: Output tensor after passing through the attention heads.
        """
        # Embed the input tokens
        x = self.embedding(x) # (batch_size, seq_length, D_MODEL)
        x = positional_encoding(x.size(1), D_MODEL) + x # Add positional encoding
        attention_weights_list = []  # To store attention weights for analysis

        for layer in range(N_LAYERS):
            # Begin Residual Attention Block
            Q_mat = self.W_Q[layer](x).view(x.size(0), x.size(1), N_HEADS, D_HEAD).transpose(1, 2) # (batch_size, N_HEADS, seq_length, D_HEAD)
            K_mat = self.W_K[layer](x).view(x.size(0), x.size(1), N_HEADS, D_HEAD).transpose(1, 2)
            V_mat = self.W_V[layer](x).view(x.size(0), x.size(1), N_HEADS, D_HEAD).transpose(1, 2)

            # Compute attention scores and weights
            attention_scores = torch.matmul(Q_mat, K_mat.transpose(-2, -1)) / np.sqrt(D_HEAD) # (batch_size, N_HEADS, seq_length, seq_length)
            mask = torch.triu(torch.ones(attention_scores.size(-1), attention_scores.size(-1), dtype=torch.bool), diagonal=1)
            attention_scores = attention_scores.masked_fill(mask, float('-inf')) # Apply causal mask
            attention_weights = self.softmax(attention_scores)
            attention_output = torch.matmul(attention_weights, V_mat) # (batch_size, N_HEADS, seq_length, D_HEAD)
            if ablate_heads is not None:
                for ablate_layer, ablate_head in ablate_heads:
                    if ablate_layer == layer:
                        attention_output[:, ablate_head, :, :] = 0
            attention_output = attention_output.transpose(1, 2).contiguous().view(x.size(0), x.size(1), D_MODEL) # (batch_size, seq_length, D_MODEL)

            # Add output back to the residual stream
            x = x + attention_output
            attention_weights_list.append(attention_weights)

        # Calculate the final output
        logits = torch.matmul(x, self.embedding.weight.T) # (batch_size, seq_length, VOCAB_SIZE)
        outputs = logits[:, -1, :] # Get the logits for the last token in the sequence
        outputs = self.softmax(outputs) # (batch_size, VOCAB_SIZE)
        return {
            "outputs": outputs,
            "logits": logits,
            "attention_weights": attention_weights_list,
        }
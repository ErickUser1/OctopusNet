"""
Nerve Ring - Lateral communication between modules

Inspired by the octopus nerve ring that connects arms to each other.
Implements cross-attention between module bottlenecks before sending to coordinator.
"""

import torch
import torch.nn as nn
import math


class NerveRing(nn.Module):
    """
    Cross-attention mechanism for lateral communication between modules.

    Each module's bottleneck h_i gets enriched with information from other modules:
    h'_i = h_i + Attention(Q=h_i, K=[h_1..h_N], V=[h_1..h_N])
    """

    def __init__(self, bottleneck_size=32, num_heads=4, dropout=0.1):
        super().__init__()
        self.bottleneck_size = bottleneck_size
        self.num_heads = num_heads
        self.head_dim = bottleneck_size // num_heads

        assert bottleneck_size % num_heads == 0, \
            f"bottleneck_size ({bottleneck_size}) must be divisible by num_heads ({num_heads})"

        # Query, Key, Value projections
        self.W_q = nn.Linear(bottleneck_size, bottleneck_size)
        self.W_k = nn.Linear(bottleneck_size, bottleneck_size)
        self.W_v = nn.Linear(bottleneck_size, bottleneck_size)

        # Output projection
        self.W_o = nn.Linear(bottleneck_size, bottleneck_size)

        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(bottleneck_size)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, bottlenecks):
        """
        Apply cross-attention between module bottlenecks.

        Args:
            bottlenecks: Tensor of shape (batch_size, num_modules, bottleneck_size)
                         Stack of all module outputs

        Returns:
            enriched: Tensor of same shape, with cross-attention applied
        """
        batch_size, num_modules, _ = bottlenecks.shape

        # Compute Q, K, V
        Q = self.W_q(bottlenecks)  # (B, N, D)
        K = self.W_k(bottlenecks)  # (B, N, D)
        V = self.W_v(bottlenecks)  # (B, N, D)

        # Reshape for multi-head attention
        # (B, N, D) -> (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        Q = Q.view(batch_size, num_modules, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_modules, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_modules, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        # (B, heads, N, head_dim) @ (B, heads, head_dim, N) -> (B, heads, N, N)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # (B, heads, N, N) @ (B, heads, N, head_dim) -> (B, heads, N, head_dim)
        attn_output = torch.matmul(attn_weights, V)

        # Reshape back
        # (B, heads, N, head_dim) -> (B, N, heads, head_dim) -> (B, N, D)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, num_modules, self.bottleneck_size)

        # Output projection
        attn_output = self.W_o(attn_output)

        # Residual connection + layer norm
        enriched = self.layer_norm(bottlenecks + self.dropout(attn_output))

        return enriched, attn_weights


class SimpleNerveRing(nn.Module):
    """
    Simplified nerve ring without multi-head attention.
    Uses single-head attention for easier interpretation.
    """

    def __init__(self, bottleneck_size=32, dropout=0.1):
        super().__init__()
        self.bottleneck_size = bottleneck_size

        # Single head attention
        self.W_q = nn.Linear(bottleneck_size, bottleneck_size)
        self.W_k = nn.Linear(bottleneck_size, bottleneck_size)
        self.W_v = nn.Linear(bottleneck_size, bottleneck_size)

        self.layer_norm = nn.LayerNorm(bottleneck_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, bottlenecks):
        """
        Args:
            bottlenecks: (batch_size, num_modules, bottleneck_size)
        Returns:
            enriched: same shape
            attn_weights: (batch_size, num_modules, num_modules)
        """
        Q = self.W_q(bottlenecks)
        K = self.W_k(bottlenecks)
        V = self.W_v(bottlenecks)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.bottleneck_size)
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply attention
        attn_output = torch.matmul(attn_weights, V)

        # Residual + norm
        enriched = self.layer_norm(bottlenecks + self.dropout(attn_output))

        return enriched, attn_weights

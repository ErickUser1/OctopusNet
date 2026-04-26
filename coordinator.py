"""
Coordinator - The "brain" of the octopus

Receives bottleneck representations from all modules (after nerve ring),
applies attention to weight their contributions, and produces final output.
Also generates feedback signals for the next batch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Coordinator(nn.Module):
    """
    Central coordinator that aggregates module outputs via attention
    and generates feedback for next batch.

    Supports different competition mechanisms inspired by GWT:
    - "soft": Standard softmax attention (default)
    - "gumbel": Gumbel-softmax with hard selection (more faithful to GWT)
    - "topk": Top-K sparse attention (only K modules contribute)
    """

    def __init__(self, num_modules=4, bottleneck_size=32, hidden_dim=256,
                 num_classes=10, use_feedback=True, competition_type="soft",
                 competition_topk=2, gumbel_tau=0.5):
        super().__init__()
        self.num_modules = num_modules
        self.bottleneck_size = bottleneck_size
        self.use_feedback = use_feedback

        # Competition mechanism
        self.competition_type = competition_type
        self.competition_topk = competition_topk
        self.gumbel_tau = gumbel_tau

        # Attention mechanism
        # Query is computed from concatenation of all bottlenecks
        self.W_q = nn.Linear(num_modules * bottleneck_size, bottleneck_size)

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Classification head
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

        # Feedback generator (one feedback vector per module)
        if use_feedback:
            # Input: aggregated representation + attention weight for each module
            self.feedback_generator = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(bottleneck_size + 1, bottleneck_size),
                    nn.Sigmoid()  # Output between 0 and 1 for modulation
                )
                for _ in range(num_modules)
            ])

        # Loss function for coordinator (trained with backprop)
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer for coordinator only
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        # Store last feedback for next batch
        self.last_feedback = None
        self.last_attention = None

    def compute_attention(self, bottlenecks):
        """
        Compute attention weights for each module using the configured
        competition mechanism.

        Args:
            bottlenecks: (batch_size, num_modules, bottleneck_size)

        Returns:
            alpha: (batch_size, num_modules) attention weights
        """
        batch_size = bottlenecks.shape[0]

        # Concatenate all bottlenecks to form query input
        concat = bottlenecks.view(batch_size, -1)  # (B, N*D)
        query = self.W_q(concat)  # (B, D)

        # Compute attention scores: query dot product with each bottleneck
        # query: (B, D) -> (B, 1, D)
        # bottlenecks: (B, N, D)
        query = query.unsqueeze(1)
        scores = torch.bmm(query, bottlenecks.transpose(1, 2))  # (B, 1, N)
        scores = scores.squeeze(1) / math.sqrt(self.bottleneck_size)  # (B, N)

        # Apply competition mechanism
        if self.competition_type == "gumbel":
            # Gumbel-softmax with hard selection (more faithful to GWT)
            # hard=True gives one-hot during forward, soft gradients during backward
            alpha = F.gumbel_softmax(scores, tau=self.gumbel_tau, hard=True, dim=-1)

        elif self.competition_type == "topk":
            # Top-K sparse attention: only top K modules contribute
            topk_vals, topk_idx = torch.topk(scores, k=self.competition_topk, dim=-1)
            # Create mask for top-k positions
            mask = torch.zeros_like(scores)
            mask.scatter_(-1, topk_idx, 1.0)
            # Apply softmax only to top-k, others get 0
            masked_scores = scores * mask + (1 - mask) * (-1e9)
            alpha = torch.softmax(masked_scores, dim=-1)

        else:  # "soft" (default)
            # Standard softmax attention
            alpha = torch.softmax(scores, dim=-1)

        return alpha

    def aggregate(self, bottlenecks, alpha):
        """
        Weighted sum of bottlenecks.

        Args:
            bottlenecks: (batch_size, num_modules, bottleneck_size)
            alpha: (batch_size, num_modules)

        Returns:
            h_agg: (batch_size, bottleneck_size)
        """
        # alpha: (B, N) -> (B, N, 1)
        alpha_expanded = alpha.unsqueeze(-1)

        # Weighted sum
        h_agg = (bottlenecks * alpha_expanded).sum(dim=1)  # (B, D)

        return h_agg

    def generate_feedback(self, h_agg, alpha):
        """
        Generate feedback vectors for each module.

        Args:
            h_agg: (batch_size, bottleneck_size) aggregated representation
            alpha: (batch_size, num_modules) attention weights

        Returns:
            feedback: list of (batch_size, bottleneck_size) tensors
        """
        if not self.use_feedback:
            return None

        feedback = []
        for i in range(self.num_modules):
            # Concatenate h_agg with attention weight for this module
            alpha_i = alpha[:, i:i+1]  # (B, 1)
            combined = torch.cat([h_agg, alpha_i], dim=-1)  # (B, D+1)

            f_i = self.feedback_generator[i](combined)  # (B, D)
            feedback.append(f_i)

        return feedback

    def forward(self, bottlenecks, return_attention=False):
        """
        Forward pass of coordinator.

        Args:
            bottlenecks: (batch_size, num_modules, bottleneck_size)
            return_attention: whether to return attention weights

        Returns:
            logits: (batch_size, num_classes)
            feedback: list of feedback vectors (if use_feedback)
            alpha: attention weights (if return_attention)
        """
        # Compute attention
        alpha = self.compute_attention(bottlenecks)
        self.last_attention = alpha.detach()

        # Aggregate
        h_agg = self.aggregate(bottlenecks, alpha)

        # Decode
        hidden = self.decoder(h_agg)

        # Classify
        logits = self.classifier(hidden)

        # Generate feedback for next batch
        feedback = self.generate_feedback(h_agg.detach(), alpha.detach())
        self.last_feedback = feedback

        if return_attention:
            return logits, feedback, alpha
        return logits, feedback

    def train_step(self, bottlenecks, labels):
        """
        Train coordinator with backprop (modules are frozen).

        Args:
            bottlenecks: (batch_size, num_modules, bottleneck_size) - detached
            labels: (batch_size,) ground truth

        Returns:
            loss, accuracy
        """
        # Ensure bottlenecks are detached (no gradients to modules)
        bottlenecks = bottlenecks.detach()

        logits, _ = self.forward(bottlenecks)

        loss = self.criterion(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compute accuracy
        preds = logits.argmax(dim=-1)
        accuracy = (preds == labels).float().mean().item()

        return loss.item(), accuracy

    def get_feedback(self):
        """Get feedback vectors from last forward pass."""
        return self.last_feedback

    def get_attention(self):
        """Get attention weights from last forward pass."""
        return self.last_attention


class AuxClassifier(nn.Module):
    """
    Local auxiliary classifier for SFF-style 100% local learning (A15b).
    Attaches to f3 feature map of each CNN module.
    Trained with CrossEntropy + detach() — no backprop to FF modules.
    """
    def __init__(self, in_channels=256, num_classes=10):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, f3):
        x = self.pool(f3).view(f3.size(0), -1)
        return self.fc(x)


class LogitCoordinator(nn.Module):
    """
    Lightweight coordinator over local logits (A15b).
    Learns attention weights over per-module logits — 100% local, no global backprop.
    Replaces the standard Coordinator when use_sff_aux=True.
    """
    def __init__(self, num_modules=4, num_classes=10):
        super().__init__()
        self.attn = nn.Linear(num_modules * num_classes, num_modules)
        self.num_modules = num_modules
        self.num_classes = num_classes

    def forward(self, logits_list):
        stacked = torch.stack(logits_list, dim=1)        # (B, M, C)
        flat = stacked.view(stacked.size(0), -1)          # (B, M*C)
        weights = torch.softmax(self.attn(flat), dim=-1)  # (B, M)
        return (stacked * weights.unsqueeze(-1)).sum(dim=1)  # (B, C)

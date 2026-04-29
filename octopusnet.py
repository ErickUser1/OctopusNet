"""
OctopusNet - Main architecture

A distributed neural network inspired by the octopus nervous system.
Combines:
- Parallel modules with local Forward-Forward learning
- Nerve ring for lateral communication
- Coordinator with attention and feedback
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import create_modules, CGCNNModule
from nerve_ring import NerveRing, SimpleNerveRing
from coordinator import Coordinator


class OctopusNet(nn.Module):
    """
    Main OctopusNet architecture.

    Architecture:
    1. Input goes to N parallel modules
    2. Each module produces bottleneck (32 values)
    3. Nerve ring: cross-attention enriches each bottleneck
    4. Coordinator: attention-weighted aggregation + classification
    5. Feedback: modulates next batch's module activations
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Create modules (the "arms")
        self.modules_list = create_modules(config)
        self.num_modules = len(self.modules_list)

        # Nerve ring (lateral communication)
        if config.use_nerve_ring:
            self.nerve_ring = NerveRing(
                bottleneck_size=config.bottleneck_size,
                num_heads=4
            )
        else:
            self.nerve_ring = None

        # Coordinator (the "brain")
        self.coordinator = Coordinator(
            num_modules=self.num_modules,
            bottleneck_size=config.bottleneck_size,
            hidden_dim=config.coordinator_hidden,
            num_classes=config.num_classes,
            use_feedback=config.use_feedback,
            competition_type=config.competition_type,
            competition_topk=config.competition_topk,
            gumbel_tau=config.gumbel_tau
        )

        # Track training state
        self.feedback_vectors = None

        # Precompute per-module input scales for fast lookup
        self._input_scales = (
            config.input_scales[:self.num_modules]
            if getattr(config, 'use_multiscale', False) and config.input_scales
            else None
        )

    def forward(self, x, return_details=False):
        """
        Forward pass through OctopusNet.

        Args:
            x: Input tensor (batch_size, channels, height, width)
            return_details: Whether to return intermediate values

        Returns:
            logits: (batch_size, num_classes)
            details: dict with bottlenecks, attention, etc. (if return_details)
        """
        batch_size = x.shape[0]

        # Apply feedback from previous batch (if available)
        if self.feedback_vectors is not None and self.config.use_feedback:
            for i, module in enumerate(self.modules_list):
                module.set_feedback(self.feedback_vectors[i])

        # Forward through all modules in parallel
        bottlenecks = []
        for i, module in enumerate(self.modules_list):
            x_i = self._scale_input(x, i)
            h_i = module(x_i)  # (batch_size, bottleneck_size)
            bottlenecks.append(h_i)

        # Stack bottlenecks: (batch_size, num_modules, bottleneck_size)
        bottlenecks = torch.stack(bottlenecks, dim=1)

        # Nerve ring: cross-attention between modules
        nerve_ring_attn = None
        if self.nerve_ring is not None:
            bottlenecks, nerve_ring_attn = self.nerve_ring(bottlenecks)

        # Coordinator: attention + aggregation + classification
        logits, feedback, coord_attn = self.coordinator(
            bottlenecks, return_attention=True
        )

        # Store feedback for next batch
        self.feedback_vectors = feedback

        if return_details:
            details = {
                'bottlenecks': bottlenecks,
                'nerve_ring_attention': nerve_ring_attn,
                'coordinator_attention': coord_attn,
                'feedback': feedback
            }
            return logits, details

        return logits

    def _scale_input(self, x, module_idx):
        """Downsample input to module's expected resolution (bilinear). No-op if multiscale off."""
        if self._input_scales is None:
            return x
        size = self._input_scales[module_idx]
        if x.shape[-1] == size:
            return x
        return F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)

    def _is_cg(self):
        """True if modules are CGCNNModule (channel grouping mode)."""
        return isinstance(self.modules_list[0], CGCNNModule)

    def train_modules_ff(self, x_pos, x_neg=None, labels=None):
        """
        Train all modules with Forward-Forward algorithm.

        Two modes:
        - Standard: x_pos=positive images, x_neg=negative images (CNNModule)
        - Channel grouping: x_pos=images, labels=ground truth (CGCNNModule, no x_neg)
        """
        losses = []
        goodness_pos = []
        goodness_neg = []

        # CG path handled externally in train.py (cg_optimizer with MultiStepLR)
        # This fallback exists only if called directly without external optimizer
        use_grouping = getattr(self.config, 'ff_channel_grouping', False)
        for i, module in enumerate(self.modules_list):
            if use_grouping and labels is not None:
                loss, g_pos, g_neg = module.train_ff(
                    self._scale_input(x_pos, i),
                    labels=labels
                )
            else:
                loss, g_pos, g_neg = module.train_ff(
                    self._scale_input(x_pos, i),
                    self._scale_input(x_neg, i)
                )
            losses.append(loss)
            goodness_pos.append(g_pos)
            goodness_neg.append(g_neg)

        return losses, goodness_pos, goodness_neg

    def train_coordinator(self, x, labels):
        """
        Train coordinator with backprop (modules frozen).
        If config.module_dropout_prob > 0, zeros a random module per batch
        with that probability — forces distributed representations (A6b).
        """
        drop_prob = getattr(self.config, 'module_dropout_prob', 0.0)

        with torch.no_grad():
            bottlenecks = [module(self._scale_input(x, i))
                           for i, module in enumerate(self.modules_list)]

            if drop_prob > 0.0 and random.random() < drop_prob:
                drop_idx = random.randint(0, self.num_modules - 1)
                bottlenecks[drop_idx] = torch.zeros_like(bottlenecks[drop_idx])

            bottlenecks = torch.stack(bottlenecks, dim=1)

            if self.nerve_ring is not None:
                bottlenecks, _ = self.nerve_ring(bottlenecks)

        loss, accuracy = self.coordinator.train_step(bottlenecks, labels)
        return loss, accuracy

    def predict(self, x):
        """
        Make predictions.

        Args:
            x: Input images

        Returns:
            predictions: (batch_size,) predicted classes
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            predictions = logits.argmax(dim=-1)
        self.train()
        return predictions

    def get_module_contributions(self, x):
        """
        Analyze which modules contribute most for given inputs.

        Returns attention weights showing each module's contribution.
        """
        self.eval()
        with torch.no_grad():
            _, details = self.forward(x, return_details=True)
        self.train()
        return details['coordinator_attention']

    def simulate_module_failure(self, module_idx):
        """
        Simulate failure of a module by zeroing its output.
        Used for resilience experiments.
        """
        # Store original forward
        original_forward = self.modules_list[module_idx].forward

        def zero_forward(x):
            batch_size = x.shape[0]
            return torch.zeros(batch_size, self.config.bottleneck_size,
                              device=x.device)

        self.modules_list[module_idx].forward = zero_forward
        return original_forward

    def restore_module(self, module_idx, original_forward):
        """Restore a module after simulated failure."""
        self.modules_list[module_idx].forward = original_forward


def _make_fourier_label_patterns(num_classes, height, width, device):
    """
    Precompute Fourier-mode label patterns (one per class).
    Each pattern is a full-image sinusoidal wave with unique orientation+frequency.
    Ensures label info visible at every spatial position — fixes one-hot CNN blindspot.
    Based on: "Training CNNs with FF" (Scientific Reports 2025).
    """
    patterns = []
    # 4 orientations x 3 frequencies = 12 configs, use first num_classes
    orientations = [0, 45, 90, 135]
    frequencies = [1, 2, 3]

    coords_y = torch.linspace(0, 2 * 3.14159, height, device=device)
    coords_x = torch.linspace(0, 2 * 3.14159, width, device=device)
    grid_y, grid_x = torch.meshgrid(coords_y, coords_x, indexing='ij')

    for idx in range(num_classes):
        angle_deg = orientations[idx % len(orientations)]
        freq = frequencies[idx // len(orientations)]
        angle_rad = angle_deg * 3.14159 / 180.0
        wave = torch.sin(freq * (grid_x * torch.cos(torch.tensor(angle_rad)) +
                                 grid_y * torch.sin(torch.tensor(angle_rad))))
        # Normalize to [0, 1]
        wave = (wave - wave.min()) / (wave.max() - wave.min() + 1e-8)
        patterns.append(wave)

    return torch.stack(patterns, dim=0)  # (num_classes, H, W)


def overlay_label_on_image(images, labels, num_classes=10, label_strength=0.5):
    """
    Embed label as Fourier sinusoidal pattern blended into image.

    strength=0.5 confirmed empirically: at res=4x4 sep=0.61, res=32x32 sep=0.07.
    All 4 multi-scale modules reach sep>0.05 in 200 batches.
    The signal scales naturally with F.interpolate — lower resolutions have
    proportionally stronger signal, which is why multi-scale FF works.
    """
    batch_size, channels, height, width = images.shape
    device = images.device

    patterns = _make_fourier_label_patterns(num_classes, height, width, device)
    label_patterns = patterns[labels].unsqueeze(1).expand(-1, channels, -1, -1)
    return images * (1.0 - label_strength) + label_patterns * label_strength


def create_negative_samples(images, labels, num_classes=10):
    """
    Create negative samples by assigning wrong labels.

    Args:
        images: (batch_size, channels, height, width)
        labels: (batch_size,) correct labels

    Returns:
        negative images with wrong labels overlaid
    """
    batch_size = images.shape[0]
    device = images.device

    # Generate random wrong labels
    wrong_labels = torch.randint(0, num_classes, (batch_size,), device=device)

    # Make sure they're actually wrong
    mask = wrong_labels == labels
    while mask.any():
        wrong_labels[mask] = torch.randint(0, num_classes, (mask.sum(),), device=device)
        mask = wrong_labels == labels

    return overlay_label_on_image(images, wrong_labels, num_classes)

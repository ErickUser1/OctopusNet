"""
OctopusNet Configuration
"""

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModuleConfig:
    """Configuration for a single module"""
    type: str = "cnn"  # "cnn", "transformer", "lstm"
    kernel_size: int = 3
    channels: List[int] = None

    def __post_init__(self):
        if self.channels is None:
            self.channels = [64, 128, 256]

@dataclass
class OctopusNetConfig:
    """Main configuration for OctopusNet"""

    # Dataset
    dataset: str = "cifar10"  # "mnist", "fashion_mnist", "cifar10", "cifar100"
    batch_size: int = 128

    # Architecture
    num_modules: int = 4
    bottleneck_size: int = 64
    num_classes: int = 10

    # Module configs (homogeneous by default)
    homogeneous: bool = True
    # All modules use kernel 3x3 — specialization comes from resolution, not kernel size.
    # Empirically confirmed: kernels 5,7,9 on same resolution are redundant (coordinator
    # ignores middle modules). Different resolutions force genuinely different information.
    kernel_sizes: List[int] = None  # [3, 3, 3, 3] — same kernel, different resolution

    # Multi-scale input: each module sees different resolution (V1/V2/V4/IT analog)
    # Default ON — empirically confirmed all 4 modules separate with Fourier+multiscale
    use_multiscale: bool = True
    input_scales: List[int] = None  # [32, 16, 8, 4] for CIFAR, [28, 14, 7, 4] for MNIST

    # Components (can be disabled for ablation)
    use_nerve_ring: bool = True
    use_feedback: bool = True

    # Competition mechanism (GWT-inspired)
    # "soft" = standard softmax (default)
    # "gumbel" = Gumbel-softmax with hard selection
    # "topk" = Top-K sparse attention
    competition_type: str = "soft"
    competition_topk: int = 2  # K value for topk competition
    gumbel_tau: float = 0.5  # Temperature for Gumbel-softmax

    # Forward-Forward
    ff_threshold: float = 2.0  # Initial value (adapts if adaptive=True)
    ff_adaptive_threshold: bool = True  # DEFAULT: adaptive threshold ON
    ff_epochs_per_layer: int = 100
    # Channel grouping (Ortiz Torres et al., arXiv:2504.21662)
    # Replaces CNNModule with CGCNNModule — no x_neg needed, goodness per class group
    ff_channel_grouping: bool = True
    cg_channels_per_group: int = 16  # total channels = num_classes * cg_channels_per_group

    # Stride conv compression (A21b): reemplaza AdaptiveAvgPool2d con Conv2d(stride=2)
    # Confirmed: 68.65% CIFAR-10, floor single 67.03%, floor doble 56.03%
    use_stride_compress: bool = True

    # Module Dropout (A21b): zero random module during coordinator training
    # p=0.7 confirmed optimal with stride conv — distributes specialization uniformly
    module_dropout_prob: float = 0.7

    # Coordinator
    coordinator_hidden: int = 256

    # Training
    learning_rate: float = 0.001
    epochs: int = 30
    device: str = "cuda"

    # Peer normalization (from loeweX repo)
    peer_normalization: float = 0.03
    momentum: float = 0.9

    def __post_init__(self):
        if self.kernel_sizes is None:
            self.kernel_sizes = [3, 3, 3, 3]

        if self.input_scales is None:
            base = 28 if self.dataset in ["mnist", "fashion_mnist"] else 32
            # [32,16,8,4] for CIFAR — each module sees half the previous resolution
            self.input_scales = [base, base // 2, base // 4, max(base // 8, 4)]

        # Set num_classes based on dataset
        if self.dataset == "cifar100":
            self.num_classes = 100
        else:
            self.num_classes = 10


# Preset configurations for experiments
def get_baseline_config():
    """Baseline: 4 CNN modules, all components enabled"""
    return OctopusNetConfig()

def get_ablation_no_nerve_ring():
    """A8: Without nerve ring"""
    return OctopusNetConfig(use_nerve_ring=False)

def get_ablation_no_feedback():
    """A7: Without feedback"""
    return OctopusNetConfig(use_feedback=False)

def get_heterogeneous_config():
    """A9: Heterogeneous modules (CNN + Transformer + LSTM)"""
    return OctopusNetConfig(homogeneous=False)

def get_gumbel_competition_config():
    """A10b: Gumbel-softmax competition (hard selection)"""
    return OctopusNetConfig(competition_type="gumbel")

def get_topk_competition_config(k=2):
    """A10c: Top-K sparse attention"""
    return OctopusNetConfig(competition_type="topk", competition_topk=k)

def get_multiscale_config(dataset="cifar10"):
    """A12b: Adaptive kernels + multi-scale input (V1/V2/V4/IT analog)"""
    return OctopusNetConfig(
        dataset=dataset,
        use_multiscale=True,
        kernel_sizes=[7, 5, 3, 3],  # fine→coarse, clamped per resolution
    )

def get_adaptive_kernels_only_config(dataset="cifar10"):
    """A12a: Adaptive kernels only, no multi-scale (control)"""
    return OctopusNetConfig(
        dataset=dataset,
        use_multiscale=False,
        kernel_sizes=[7, 5, 3, 3],
    )

def get_channel_grouping_config(dataset="cifar10"):
    """A18b: Channel grouping FF — single forward pass, no x_neg generation"""
    return OctopusNetConfig(
        dataset=dataset,
        ff_channel_grouping=True,
        cg_channels_per_group=16,
    )

def get_a6b_config(dataset="cifar10"):
    """A6b: Channel Grouping + Module Dropout p=0.5 (no stride conv).
    64.34% accuracy, single-failure floor 61.12%, double-failure floor 52.87%.
    """
    return OctopusNetConfig(
        dataset=dataset,
        ff_channel_grouping=True,
        use_stride_compress=False,
        cg_channels_per_group=16,
        module_dropout_prob=0.5,
        epochs=30,
        bottleneck_size=64,
    )

def get_a21b_config(dataset="cifar10"):
    """A21b: Stride Conv + Module Dropout p=0.7 — best overall config.
    68.65% accuracy, single-failure floor 67.03%, double-failure floor 56.03%.
    """
    return OctopusNetConfig(
        dataset=dataset,
        ff_channel_grouping=True,
        use_stride_compress=True,
        cg_channels_per_group=16,
        module_dropout_prob=0.7,
        epochs=30,
        bottleneck_size=64,
    )

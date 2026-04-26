"""
Data loading utilities for OctopusNet experiments.
"""

import torch
import math
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(dataset_name):
    """Get appropriate transforms for each dataset."""

    if dataset_name in ["mnist", "fashion_mnist"]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    elif dataset_name in ["cifar10", "cifar100"]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            )
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return transform


def get_dataloaders(config):
    """
    Get train and test dataloaders for the specified dataset.

    Args:
        config: OctopusNetConfig

    Returns:
        train_loader, test_loader
    """
    transform = get_transforms(config.dataset)

    if config.dataset == "mnist":
        train_dataset = datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
    elif config.dataset == "fashion_mnist":
        train_dataset = datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=transform
        )
    elif config.dataset == "cifar10":
        train_dataset = datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
    elif config.dataset == "cifar100":
        train_dataset = datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, test_loader


# ── Label embedding ────────────────────────────────────────────────────────────

_FOURIER_CACHE = {}

def _make_fourier_patterns(num_classes, height, width, device):
    key = (num_classes, height, width, str(device))
    if key in _FOURIER_CACHE:
        return _FOURIER_CACHE[key]
    orientations = [0, 45, 90, 135]
    frequencies  = [1, 2, 3]
    cy = torch.linspace(0, 2*math.pi, height, device=device)
    cx = torch.linspace(0, 2*math.pi, width,  device=device)
    gy, gx = torch.meshgrid(cy, cx, indexing='ij')
    patterns = []
    for idx in range(num_classes):
        angle = math.radians(orientations[idx % len(orientations)])
        freq  = frequencies[idx // len(orientations)]
        wave  = torch.sin(freq * (gx*math.cos(angle) + gy*math.sin(angle)))
        wave  = (wave - wave.min()) / (wave.max() - wave.min() + 1e-8)
        patterns.append(wave)
    result = torch.stack(patterns, dim=0)  # (num_classes, H, W)
    _FOURIER_CACHE[key] = result
    return result


def overlay_label_on_image(images, labels, num_classes=10, label_strength=0.5):
    """
    Fourier label embedding: blend sinusoidal class pattern into image.

    Each class gets a unique orientation+frequency sinusoid visible at every
    spatial position. At strength=0.5, the signal survives F.interpolate
    downscaling to 4x4 and gives enough g_pos/g_neg separation for FF to learn.

    WHY strength=0.5: tested empirically — at res=4x4 sep reaches 0.61,
    at res=32x32 sep reaches 0.07. All 4 multi-scale modules separate.
    """
    B, C, H, W = images.shape
    patterns = _make_fourier_patterns(num_classes, H, W, images.device)
    pat = patterns[labels].unsqueeze(1).expand(-1, C, -1, -1)
    return images * (1.0 - label_strength) + pat * label_strength


def create_negative_samples(images, labels, num_classes=10):
    """Create negative samples by assigning wrong Fourier label patterns."""
    wrong = torch.randint(0, num_classes, labels.shape, device=labels.device)
    mask = wrong == labels
    while mask.any():
        wrong[mask] = torch.randint(0, num_classes, (mask.sum(),), device=labels.device)
        mask = wrong == labels
    return overlay_label_on_image(images, wrong, num_classes)


def get_dataset_info(dataset_name):
    """Get dataset-specific information."""
    info = {
        "mnist": {
            "num_classes": 10,
            "input_channels": 1,
            "input_size": 28,
            "num_train": 60000,
            "num_test": 10000
        },
        "fashion_mnist": {
            "num_classes": 10,
            "input_channels": 1,
            "input_size": 28,
            "num_train": 60000,
            "num_test": 10000
        },
        "cifar10": {
            "num_classes": 10,
            "input_channels": 3,
            "input_size": 32,
            "num_train": 50000,
            "num_test": 10000
        },
        "cifar100": {
            "num_classes": 100,
            "input_channels": 3,
            "input_size": 32,
            "num_train": 50000,
            "num_test": 10000
        }
    }
    return info.get(dataset_name, None)

"""
Training script for OctopusNet.

Two training modes:
1. Standard (default): FF modules + backprop coordinator
2. SFF mode (--use_sff): FF modules + AuxClassifier per module + LogitCoordinator
   Fully local — no global backprop anywhere. Best accuracy: 53.16% on CIFAR-10.

Usage:
    python train.py                          # standard mode
    python train.py --use_sff               # 100% local SFF mode
    python train.py --dataset cifar100      # different dataset
    python train.py --epochs 100            # more epochs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json
import argparse
from datetime import datetime

from config import OctopusNetConfig
from octopusnet import OctopusNet, overlay_label_on_image, create_negative_samples
from coordinator import AuxClassifier, LogitCoordinator
from data import get_dataloaders


def train_epoch_ff(model, train_loader, config, epoch):
    """Train modules with Forward-Forward for one epoch."""
    model.train()
    total_losses = [0.0] * model.num_modules
    total_g_pos = [0.0] * model.num_modules
    total_g_neg = [0.0] * model.num_modules
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [FF]")
    for images, labels in pbar:
        images = images.to(config.device)
        labels = labels.to(config.device)

        if getattr(config, 'ff_channel_grouping', False):
            x_pos = overlay_label_on_image(images, labels, config.num_classes)
            losses, g_pos, g_neg = model.train_modules_ff(x_pos, labels=labels)
        else:
            x_pos = overlay_label_on_image(images, labels, config.num_classes)
            x_neg = create_negative_samples(images, labels, config.num_classes)
            losses, g_pos, g_neg = model.train_modules_ff(x_pos, x_neg)

        for i in range(model.num_modules):
            total_losses[i] += losses[i]
            total_g_pos[i] += g_pos[i]
            total_g_neg[i] += g_neg[i]
        num_batches += 1

        avg_loss = sum(losses) / len(losses)
        pbar.set_postfix({'avg_ff_loss': f'{avg_loss:.4f}'})

    return (
        [l / num_batches for l in total_losses],
        [g / num_batches for g in total_g_pos],
        [g / num_batches for g in total_g_neg],
    )


def train_epoch_coordinator(model, train_loader, config, epoch):
    """Train coordinator with backprop for one epoch (standard mode)."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Coord]")
    for images, labels in pbar:
        images = images.to(config.device)
        labels = labels.to(config.device)

        loss, acc = model.train_coordinator(images, labels)
        total_loss += loss
        total_acc += acc
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss:.4f}', 'acc': f'{acc:.4f}'})

    return total_loss / num_batches, total_acc / num_batches


def train_epoch_sff(model, aux_classifiers, logit_coord, aux_optimizer,
                    train_loader, config, epoch):
    """
    Train AuxClassifiers + LogitCoordinator for one epoch (SFF mode).

    100% local — no global backprop. Each module has its own AuxClassifier
    trained with CrossEntropy. LogitCoordinator learns attention over logits.
    detach() prevents gradients from flowing back to FF modules.

    Based on: Krutsylo (arXiv:2501.03176, 2025) — SFF applied to OctopusNet.
    Best result: 53.16% on CIFAR-10 (vs 52.50% with backprop coordinator).
    """
    model.train()
    aux_classifiers.train()
    logit_coord.train()

    criterion = nn.CrossEntropyLoss()
    resolutions = config.input_scales if getattr(config, 'use_multiscale', False) else [32] * model.num_modules
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [SFF]")
    for images, labels in pbar:
        images = images.to(config.device)
        labels = labels.to(config.device)

        # FF training (internal optimizer per module)
        x_pos = overlay_label_on_image(images, labels, config.num_classes)
        x_neg = create_negative_samples(images, labels, config.num_classes)
        model.train_modules_ff(x_pos, x_neg)

        # AuxClassifier + LogitCoordinator (local, no backprop to FF modules)
        aux_optimizer.zero_grad()
        logits_list = []
        loss_aux = 0.0

        for i, (module, aux_cls) in enumerate(zip(model.modules_list, aux_classifiers)):
            res = resolutions[i]
            x_i = F.interpolate(images, size=(res, res), mode='bilinear', align_corners=False)
            _, _, f3 = module._conv_features(x_i)
            logits_i = aux_cls(f3.detach())  # detach: no backprop to FF modules
            logits_list.append(logits_i)
            loss_aux += criterion(logits_i, labels)

        logits_coord = logit_coord([l.detach() for l in logits_list])
        loss_coord = criterion(logits_coord, labels)
        (loss_aux + loss_coord).backward()
        aux_optimizer.step()

        acc = (logits_coord.argmax(1) == labels).float().mean().item()
        total_loss += (loss_aux + loss_coord).item()
        total_acc += acc
        num_batches += 1
        pbar.set_postfix({'loss': f'{(loss_aux+loss_coord).item():.4f}', 'acc': f'{acc:.4f}'})

    return total_loss / num_batches, total_acc / num_batches


def evaluate(model, test_loader, config,
             aux_classifiers=None, logit_coord=None):
    """
    Evaluate model. If aux_classifiers provided, uses SFF inference.
    Otherwise uses standard coordinator forward pass.
    """
    model.eval()
    if aux_classifiers is not None:
        aux_classifiers.eval()
        logit_coord.eval()

    correct = 0
    total = 0
    resolutions = config.input_scales if getattr(config, 'use_multiscale', False) else [32] * model.num_modules

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.device)
            labels = labels.to(config.device)

            if aux_classifiers is not None:
                # SFF inference: logit coordinator
                logits_list = []
                for i, (module, aux_cls) in enumerate(zip(model.modules_list, aux_classifiers)):
                    res = resolutions[i]
                    x_i = F.interpolate(images, size=(res, res), mode='bilinear', align_corners=False)
                    _, _, f3 = module._conv_features(x_i)
                    logits_list.append(aux_cls(f3))
                preds = logit_coord(logits_list).argmax(1)
            else:
                preds = model.predict(images)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def train(config, use_sff=False, seed=42):
    """
    Full training loop for OctopusNet.

    Args:
        config: OctopusNetConfig
        use_sff: if True, use 100% local SFF mode (AuxClassifier + LogitCoordinator)
        seed: random seed for reproducibility
    """
    # Reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    print(f"Training OctopusNet on {config.dataset}")
    print(f"Mode: {'SFF 100% local' if use_sff else 'Standard (FF + backprop coord)'}")
    print(f"Config: {config}")

    train_loader, test_loader = get_dataloaders(config)
    model = OctopusNet(config).to(config.device)

    # SFF components (only if use_sff=True)
    aux_classifiers = None
    logit_coord = None
    aux_optimizer = None

    if use_sff:
        aux_classifiers = nn.ModuleList([
            AuxClassifier(in_channels=256, num_classes=config.num_classes).to(config.device)
            for _ in range(model.num_modules)
        ])
        logit_coord = LogitCoordinator(
            num_modules=model.num_modules,
            num_classes=config.num_classes
        ).to(config.device)
        aux_optimizer = torch.optim.Adam(
            list(aux_classifiers.parameters()) + list(logit_coord.parameters()),
            lr=0.001
        )

    history = {
        'ff_losses': [], 'coord_losses': [],
        'coord_train_acc': [], 'test_acc': [],
        'module_goodness_pos': [], 'module_goodness_neg': []
    }
    best_acc = 0.0

    for epoch in range(1, config.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config.epochs}")
        print(f"{'='*50}")

        # Phase 1: FF training (always)
        ff_losses, g_pos, g_neg = train_epoch_ff(model, train_loader, config, epoch)
        print(f"FF Losses: {[f'{l:.4f}' for l in ff_losses]}")
        print(f"Goodness pos: {[f'{g:.4f}' for g in g_pos]}")
        print(f"Goodness neg: {[f'{g:.4f}' for g in g_neg]}")

        # Phase 2: Coordinator training
        if use_sff:
            coord_loss, train_acc = train_epoch_sff(
                model, aux_classifiers, logit_coord, aux_optimizer,
                train_loader, config, epoch
            )
        else:
            coord_loss, train_acc = train_epoch_coordinator(
                model, train_loader, config, epoch
            )

        print(f"Coordinator loss: {coord_loss:.4f} | Train acc: {train_acc:.4f}")

        # Evaluate
        test_acc = evaluate(model, test_loader, config, aux_classifiers, logit_coord)
        print(f"Test accuracy: {test_acc:.4f}")

        history['ff_losses'].append(ff_losses)
        history['coord_losses'].append(coord_loss)
        history['coord_train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['module_goodness_pos'].append(g_pos)
        history['module_goodness_neg'].append(g_neg)

        if test_acc > best_acc:
            best_acc = test_acc
            ckpt = {'model': model.state_dict(), 'epoch': epoch, 'acc': best_acc}
            if use_sff:
                ckpt['aux'] = aux_classifiers.state_dict()
                ckpt['coord'] = logit_coord.state_dict()
            torch.save(ckpt, 'best_model.pt')
            print(f"New best: {best_acc:.4f}")

    print(f"\nTraining complete! Best: {best_acc:.4f}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'history_{timestamp}.json', 'w') as f:
        json.dump(history, f, indent=2)

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OctopusNet")
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'mnist'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--bottleneck', type=int, default=64)
    parser.add_argument('--use_sff', action='store_true',
                        help='Use 100%% local SFF mode (AuxClassifier + LogitCoordinator). '
                             'Best accuracy: 53.16%% on CIFAR-10.')
    parser.add_argument('--no_multiscale', action='store_true',
                        help='Disable multiscale input (each module sees same resolution). '
                             'Default: multiscale ON (each module gets different resolution).')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    config = OctopusNetConfig(
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        bottleneck_size=args.bottleneck,
        device=args.device,
        use_multiscale=not args.no_multiscale,
    )

    model, history = train(config, use_sff=args.use_sff, seed=args.seed)

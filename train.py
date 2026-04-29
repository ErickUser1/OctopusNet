"""
Training script for OctopusNet.

Three training modes:
1. Standard (default): FF modules + backprop coordinator. ~52.75% CIFAR-10.
2. SFF mode (--use_sff): FF modules + AuxClassifier + LogitCoordinator.
   Fully local — no global backprop anywhere. ~53.16% CIFAR-10.
3. A6b mode (--channel_grouping --module_dropout 0.5): CGCNNModule + Module Dropout.
   Best overall — 64.34% CIFAR-10, single-failure floor 61.12%.

Usage:
    python train.py                                          # standard mode
    python train.py --use_sff                               # 100% local SFF
    python train.py --channel_grouping --module_dropout 0.5 # A6b (recommended)
    python train.py --dataset cifar100 --epochs 50
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import json
import argparse
from datetime import datetime

from config import OctopusNetConfig
from modules import CGCNNModule
from octopusnet import OctopusNet, overlay_label_on_image, create_negative_samples
from coordinator import AuxClassifier, LogitCoordinator
from data import get_dataloaders


def _is_cg_model(model):
    return isinstance(model.modules_list[0], CGCNNModule)


def train_epoch_ff(model, train_loader, config, epoch, cg_optimizer=None):
    """
    Train modules with Forward-Forward for one epoch.
    CG mode: uses cg_optimizer (external, with MultiStepLR) instead of
    per-module Adam — lets the scheduler control lr across epochs.
    """
    model.train()
    total_losses = [0.0] * model.num_modules
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [FF]")
    for images, labels in pbar:
        images = images.to(config.device)
        labels = labels.to(config.device)

        if _is_cg_model(model):
            cg_optimizer.zero_grad()
            ff_loss = sum(mod.ff_loss(images, labels) for mod in model.modules_list)
            ff_loss.backward()
            cg_optimizer.step()
            per_mod = ff_loss.item() / model.num_modules
            losses = [per_mod] * model.num_modules
        else:
            x_pos = overlay_label_on_image(images, labels, config.num_classes)
            x_neg = create_negative_samples(images, labels, config.num_classes)
            losses, _, _ = model.train_modules_ff(x_pos, x_neg)

        for i in range(model.num_modules):
            total_losses[i] += losses[i]
        num_batches += 1
        pbar.set_postfix({'avg_ff_loss': f'{sum(losses)/len(losses):.4f}'})

    return [l / num_batches for l in total_losses]


def train_epoch_coordinator(model, train_loader, config, epoch, coord_optimizer=None):
    """
    Train coordinator with backprop for one epoch.
    Module dropout (if configured) is applied inside model.train_coordinator().
    coord_optimizer is unused here — MultiStepLR steps it externally in train().
    """
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

    Mode is determined by config flags:
        ff_channel_grouping=True  → CGCNNModule (A18b/A6b)
        module_dropout_prob=0.5   → Module Dropout during coordinator training (A6b)
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    cg_mode = getattr(config, 'ff_channel_grouping', False)
    mod_drop = getattr(config, 'module_dropout_prob', 0.0)

    mode_str = 'SFF 100% local' if use_sff else (
        f'CG + ModDrop(p={mod_drop})' if cg_mode and mod_drop > 0 else
        'CG (channel grouping)' if cg_mode else
        'Standard (FF + backprop coord)'
    )
    print(f"Training OctopusNet on {config.dataset} — {mode_str}")

    train_loader, test_loader = get_dataloaders(config)
    model = OctopusNet(config).to(config.device)

    # Schedulers for CG mode (Ortiz Torres prescription)
    cg_optimizer = None
    coord_optimizer = None
    cg_sched = None
    coord_sched = None
    if cg_mode:
        cg_optimizer = torch.optim.Adam(model.modules_list.parameters(), lr=1e-3)
        coord_params = (
            list(model.nerve_ring.parameters()) if model.nerve_ring else []
        ) + list(model.coordinator.parameters())
        coord_optimizer = torch.optim.Adam(coord_params, lr=1e-3)
        # Replace internal coordinator optimizer so MultiStepLR controls it
        model.coordinator.optimizer = coord_optimizer
        cg_sched = MultiStepLR(cg_optimizer, milestones=[10, 20, 27], gamma=0.1)
        coord_sched = MultiStepLR(coord_optimizer, milestones=[10, 20, 27], gamma=0.1)

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

    history = {'ff_losses': [], 'coord_losses': [], 'coord_train_acc': [], 'test_acc': []}
    best_acc = 0.0

    for epoch in range(1, config.epochs + 1):
        print(f"\n{'='*50}\nEpoch {epoch}/{config.epochs}\n{'='*50}")

        # Phase 1: FF training
        ff_losses = train_epoch_ff(model, train_loader, config, epoch, cg_optimizer)
        print(f"FF Losses: {[f'{l:.4f}' for l in ff_losses]}")

        # Phase 2: Coordinator training
        if use_sff:
            coord_loss, train_acc = train_epoch_sff(
                model, aux_classifiers, logit_coord, aux_optimizer,
                train_loader, config, epoch
            )
        else:
            coord_loss, train_acc = train_epoch_coordinator(
                model, train_loader, config, epoch, coord_optimizer
            )

        if cg_sched:
            cg_sched.step()
            coord_sched.step()

        print(f"Coordinator loss: {coord_loss:.4f} | Train acc: {train_acc:.4f}")

        test_acc = evaluate(model, test_loader, config, aux_classifiers, logit_coord)
        print(f"Test accuracy: {test_acc:.4f}")

        history['ff_losses'].append(ff_losses)
        history['coord_losses'].append(coord_loss)
        history['coord_train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            ckpt = {'model': model.state_dict(), 'epoch': epoch, 'acc': best_acc}
            if use_sff:
                ckpt['aux'] = aux_classifiers.state_dict()
                ckpt['coord'] = logit_coord.state_dict()
            torch.save(ckpt, 'best_model.pt')
            print(f"New best: {best_acc:.4f}")

    print(f"\nTraining complete! Best: {best_acc:.4f}")
    with open(f'history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
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
                        help='Disable multiscale input. Default: ON.')
    parser.add_argument('--channel_grouping', action='store_true',
                        help='Use CGCNNModule (Ortiz Torres et al.). '
                             'Best accuracy: 64.34%% CIFAR-10.')
    parser.add_argument('--module_dropout', type=float, default=0.0,
                        help='Module dropout probability during coordinator training. '
                             'p=0.5 recommended with --channel_grouping (A6b config). '
                             'Raises single-failure floor from 41.47%% to 61.12%%.')
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
        ff_channel_grouping=args.channel_grouping,
        module_dropout_prob=args.module_dropout,
    )

    model, history = train(config, use_sff=args.use_sff, seed=args.seed)

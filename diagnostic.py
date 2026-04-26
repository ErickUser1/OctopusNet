"""
OctopusNet Diagnostic Script
Runs 3 epochs and reports detailed training health metrics.
Run this BEFORE any long training to verify FF is learning correctly.

Key questions answered:
  1. Is g_pos > g_neg and is the gap growing? (FF health)
  2. Are modules learning differently? (specialization signal)
  3. Is coordinator accuracy improving? (end-to-end signal)
  4. Are there NaN/Inf values? (numerical stability)
  5. What is the effective receptive field problem? (kernel audit)
"""

import torch
import torch.nn as nn
import math

# ── Config ────────────────────────────────────────────────────────────────────

DATASET     = "cifar10"   # "mnist" is faster for quick check
EPOCHS      = 3
BATCH_SIZE  = 128
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_MODULES = 4
KERNELS     = [3, 5, 7, 9]

# ── Helpers ───────────────────────────────────────────────────────────────────

def sep(char="─", n=60): print(char * n)

def check_tensor(t, name):
    """Flag NaN/Inf in any tensor."""
    if torch.isnan(t).any():
        print(f"  ⚠ NaN in {name}")
        return False
    if torch.isinf(t).any():
        print(f"  ⚠ Inf in {name}")
        return False
    return True

def effective_receptive_field(kernel_size, num_layers):
    """RF grows as: RF = 1 + (k-1)*num_layers for same-stride convs."""
    return 1 + (kernel_size - 1) * num_layers

# ── Kernel audit ──────────────────────────────────────────────────────────────

def audit_kernels(kernels, image_size=32, num_layers=3):
    sep()
    print("KERNEL AUDIT")
    sep()
    print(f"  Image size: {image_size}×{image_size}")
    print(f"  Layers per module: {num_layers}")
    print()
    for i, k in enumerate(kernels):
        rf = effective_receptive_field(k, num_layers)
        coverage = rf / image_size * 100
        flag = "⚠ RF > IMAGE" if rf >= image_size else "OK"
        print(f"  Module {i+1} kernel={k}×{k}  RF={rf}×{rf}  "
              f"coverage={coverage:.0f}%  {flag}")
    print()

# ── Main diagnostic ───────────────────────────────────────────────────────────

def run_diagnostic():
    print()
    sep("═")
    print("OCTOPUSNET DIAGNOSTIC")
    sep("═")
    print(f"  Device : {DEVICE}")
    print(f"  Dataset: {DATASET}")
    print(f"  Epochs : {EPOCHS}")
    print()

    # Kernel audit before loading anything
    img_size = 28 if DATASET in ["mnist", "fashion_mnist"] else 32
    audit_kernels(KERNELS, image_size=img_size)

    # Import here so audit runs even if torch import fails
    from config import OctopusNetConfig
    from octopusnet import OctopusNet, overlay_label_on_image, create_negative_samples
    from data import get_dataloaders

    config = OctopusNetConfig(
        dataset=DATASET,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        num_modules=NUM_MODULES,
        kernel_sizes=KERNELS,
        device=DEVICE,
        ff_adaptive_threshold=True,
    )

    train_loader, test_loader = get_dataloaders(config)
    model = OctopusNet(config).to(DEVICE)

    # Count parameters
    sep()
    print("MODEL SIZE")
    sep()
    total = sum(p.numel() for p in model.parameters())
    for i, mod in enumerate(model.modules_list):
        n = sum(p.numel() for p in mod.parameters())
        print(f"  Module {i+1} (k={KERNELS[i]}): {n:,} params")
    coord_n = sum(p.numel() for p in model.coordinator.parameters())
    print(f"  Coordinator       : {coord_n:,} params")
    if model.nerve_ring:
        nr_n = sum(p.numel() for p in model.nerve_ring.parameters())
        print(f"  Nerve ring        : {nr_n:,} params")
    print(f"  TOTAL             : {total:,} params")
    print()

    # Per-epoch diagnostics
    history = {
        'g_pos': [[] for _ in range(NUM_MODULES)],
        'g_neg': [[] for _ in range(NUM_MODULES)],
        'sep':   [[] for _ in range(NUM_MODULES)],
        'ff_loss': [[] for _ in range(NUM_MODULES)],
        'coord_loss': [],
        'train_acc': [],
        'test_acc': [],
    }

    for epoch in range(1, EPOCHS + 1):
        sep("═")
        print(f"EPOCH {epoch}/{EPOCHS}")
        sep("═")

        # ── FF training ───────────────────────────────────────────────────────
        model.train()
        epoch_g_pos  = [0.0] * NUM_MODULES
        epoch_g_neg  = [0.0] * NUM_MODULES
        epoch_ff_loss = [0.0] * NUM_MODULES
        num_batches = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            x_pos = overlay_label_on_image(images, labels, config.num_classes)
            x_neg = create_negative_samples(images, labels, config.num_classes)

            losses, g_pos, g_neg = model.train_modules_ff(x_pos, x_neg)

            for i in range(NUM_MODULES):
                epoch_g_pos[i]   += g_pos[i]
                epoch_g_neg[i]   += g_neg[i]
                epoch_ff_loss[i] += losses[i]
            num_batches += 1

        # ── FF summary ────────────────────────────────────────────────────────
        print("\nFORWARD-FORWARD HEALTH")
        sep()
        print(f"  {'Module':<10} {'g_pos':>8} {'g_neg':>8} {'sep':>8} {'loss':>8}  status")
        sep()

        all_healthy = True
        for i in range(NUM_MODULES):
            gp  = epoch_g_pos[i]  / num_batches
            gn  = epoch_g_neg[i]  / num_batches
            s   = gp - gn
            fl  = epoch_ff_loss[i] / num_batches

            history['g_pos'][i].append(gp)
            history['g_neg'][i].append(gn)
            history['sep'][i].append(s)
            history['ff_loss'][i].append(fl)

            if s > 0.1:
                status = "OK - separating"
            elif s > 0:
                status = "WEAK - barely separating"
            else:
                status = "BAD - not separating"
                all_healthy = False

            print(f"  Module {i+1} (k={KERNELS[i]})"
                  f"  {gp:>8.3f} {gn:>8.3f} {s:>8.3f} {fl:>8.4f}  {status}")

        if not all_healthy:
            print("\n  ⚠ Some modules not separating pos/neg goodness.")
            print("    FF is not learning. Check: label embedding, threshold, lr.")

        # ── Coordinator training ──────────────────────────────────────────────
        model.train()
        total_coord_loss = 0.0
        total_coord_acc  = 0.0
        num_batches_coord = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            loss, acc = model.train_coordinator(images, labels)
            total_coord_loss += loss
            total_coord_acc  += acc
            num_batches_coord += 1

        avg_coord_loss = total_coord_loss / num_batches_coord
        avg_train_acc  = total_coord_acc  / num_batches_coord

        # ── Evaluation ────────────────────────────────────────────────────────
        model.eval()
        correct = 0
        total   = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                preds = model.predict(images)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        test_acc = correct / total

        history['coord_loss'].append(avg_coord_loss)
        history['train_acc'].append(avg_train_acc)
        history['test_acc'].append(test_acc)

        print(f"\nCOORDINATOR")
        sep()
        print(f"  Loss      : {avg_coord_loss:.4f}")
        print(f"  Train acc : {avg_train_acc*100:.2f}%")
        print(f"  Test acc  : {test_acc*100:.2f}%")

        # ── Attention weights ─────────────────────────────────────────────────
        sample_images = next(iter(test_loader))[0][:16].to(DEVICE)
        with torch.no_grad():
            attn = model.get_module_contributions(sample_images)  # (16, N)
        attn_mean = attn.mean(dim=0)

        print(f"\nATTENTION WEIGHTS (mean over 16 samples)")
        sep()
        for i in range(NUM_MODULES):
            bar = "█" * int(attn_mean[i].item() * 40)
            print(f"  Module {i+1} (k={KERNELS[i]}): {attn_mean[i].item():.3f}  {bar}")

        # Check specialization — if all weights ~equal, no specialization
        attn_std = attn_mean.std().item()
        if attn_std < 0.02:
            print("  ⚠ Attention weights nearly uniform — no specialization yet")
        else:
            print(f"  OK - std={attn_std:.3f}, modules differentiating")

    # ── Trend analysis ────────────────────────────────────────────────────────
    sep("═")
    print("TREND ANALYSIS (is training improving?)")
    sep("═")

    print("\nGoodness separation per module across epochs:")
    for i in range(NUM_MODULES):
        seps = history['sep'][i]
        trend = "↑ growing" if seps[-1] > seps[0] else "↓ shrinking or flat"
        vals = "  ".join(f"{s:.3f}" for s in seps)
        print(f"  Module {i+1}: {vals}  {trend}")

    print(f"\nTest accuracy across epochs:")
    accs = history['test_acc']
    vals = "  ".join(f"{a*100:.2f}%" for a in accs)
    trend = "↑ improving" if accs[-1] > accs[0] else "↓ not improving"
    print(f"  {vals}  {trend}")

    # ── Final verdict ─────────────────────────────────────────────────────────
    sep("═")
    print("VERDICT")
    sep("═")

    issues = []

    # Check RF problem
    for i, k in enumerate(KERNELS):
        rf = effective_receptive_field(k, 3)
        if rf >= img_size:
            issues.append(f"Module {i+1} kernel={k} RF={rf} >= image_size={img_size} "
                         f"— consider use_multiscale=True or smaller kernel")

    # Check FF health
    for i in range(NUM_MODULES):
        final_sep = history['sep'][i][-1]
        if final_sep <= 0:
            issues.append(f"Module {i+1} FF not separating (sep={final_sep:.3f}) "
                         f"— FF not learning, check threshold and label embedding")
        elif final_sep < 0.05:
            issues.append(f"Module {i+1} FF weakly separating (sep={final_sep:.3f}) "
                         f"— consider lower lr or more epochs before coordinator")

    # Check accuracy trend
    if accs[-1] <= accs[0] and EPOCHS > 1:
        issues.append("Test accuracy not improving — coordinator may be overfitting "
                     "to poor module representations")

    if not issues:
        print("  All checks passed. Safe to run full training.")
    else:
        print(f"  {len(issues)} issue(s) found:\n")
        for issue in issues:
            print(f"  ✗ {issue}\n")

    print()
    return history


if __name__ == "__main__":
    history = run_diagnostic()

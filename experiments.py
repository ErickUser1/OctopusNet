"""
Experiments for OctopusNet paper.

Implements all ablation studies and comparisons described in the paper.
"""

import torch
import json
from datetime import datetime
from tqdm import tqdm

from config import (
    OctopusNetConfig,
    get_baseline_config,
    get_ablation_no_nerve_ring,
    get_ablation_no_feedback,
    get_gumbel_competition_config,
    get_topk_competition_config,
    get_multiscale_config,
    get_adaptive_kernels_only_config,
    get_channel_grouping_config,
)
from octopusnet import OctopusNet
from train import train, evaluate
from data import get_dataloaders


def run_experiment(name, config, num_runs=3):
    """
    Run an experiment multiple times and average results.
    """
    print(f"\n{'#'*60}")
    print(f"# Experiment: {name}")
    print(f"{'#'*60}")

    results = {
        'name': name,
        'config': str(config),
        'runs': []
    }

    for run in range(1, num_runs + 1):
        print(f"\n--- Run {run}/{num_runs} ---")
        torch.manual_seed(run * 42)

        model, history = train(config)
        best_acc = max(history['test_acc'])

        results['runs'].append({
            'best_acc': best_acc,
            'final_acc': history['test_acc'][-1],
            'history': history
        })

    # Compute averages
    avg_best = sum(r['best_acc'] for r in results['runs']) / num_runs
    avg_final = sum(r['final_acc'] for r in results['runs']) / num_runs

    results['avg_best_acc'] = avg_best
    results['avg_final_acc'] = avg_final

    print(f"\n{'='*40}")
    print(f"Experiment: {name}")
    print(f"Average best accuracy: {avg_best:.4f}")
    print(f"Average final accuracy: {avg_final:.4f}")
    print(f"{'='*40}")

    return results


def experiment_A1_num_modules(dataset="cifar10"):
    """A1: Vary number of modules (2, 4, 8, 16)"""
    results = []
    for n in [2, 4, 8, 16]:
        config = OctopusNetConfig(
            dataset=dataset,
            num_modules=n,
            kernel_sizes=[3, 5, 7, 9, 11, 13, 15, 17][:n],
            epochs=50,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        result = run_experiment(f"A1_modules_{n}", config, num_runs=3)
        results.append(result)
    return results


def experiment_A2_bottleneck_size(dataset="cifar10"):
    """A2: Vary bottleneck size (8, 16, 32, 64, 128)"""
    results = []
    for b in [8, 16, 32, 64, 128]:
        config = OctopusNetConfig(
            dataset=dataset,
            bottleneck_size=b,
            epochs=50,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        result = run_experiment(f"A2_bottleneck_{b}", config, num_runs=3)
        results.append(result)
    return results


def experiment_A6_resilience(dataset="cifar10"):
    """A6: Test resilience by disabling modules one by one"""
    print("\n" + "#"*60)
    print("# Experiment A6: Resilience")
    print("#"*60)

    config = OctopusNetConfig(
        dataset=dataset,
        epochs=50,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Train full model first
    model, _ = train(config)
    _, test_loader = get_dataloaders(config)

    results = {
        'name': 'A6_resilience',
        'full_accuracy': evaluate(model, test_loader, config),
        'degradation': []
    }

    print(f"Full model accuracy: {results['full_accuracy']:.4f}")

    # Test with each module disabled
    for i in range(model.num_modules):
        original = model.simulate_module_failure(i)
        acc = evaluate(model, test_loader, config)
        model.restore_module(i, original)

        results['degradation'].append({
            'disabled_module': i,
            'accuracy': acc,
            'drop': results['full_accuracy'] - acc
        })
        print(f"Module {i} disabled: accuracy = {acc:.4f} (drop: {results['full_accuracy'] - acc:.4f})")

    # Test with multiple modules disabled
    for n_disabled in [2, 3]:
        if n_disabled >= model.num_modules:
            continue

        originals = []
        for i in range(n_disabled):
            originals.append((i, model.simulate_module_failure(i)))

        acc = evaluate(model, test_loader, config)

        for i, orig in originals:
            model.restore_module(i, orig)

        results['degradation'].append({
            'disabled_modules': list(range(n_disabled)),
            'accuracy': acc,
            'drop': results['full_accuracy'] - acc
        })
        print(f"Modules 0-{n_disabled-1} disabled: accuracy = {acc:.4f}")

    return results


def experiment_A7_feedback(dataset="cifar10"):
    """A7: Compare with and without feedback"""
    results = []

    # With feedback (baseline)
    config_with = get_baseline_config()
    config_with.dataset = dataset
    config_with.device = "cuda" if torch.cuda.is_available() else "cpu"
    result = run_experiment("A7_with_feedback", config_with, num_runs=3)
    results.append(result)

    # Without feedback
    config_without = get_ablation_no_feedback()
    config_without.dataset = dataset
    config_without.device = "cuda" if torch.cuda.is_available() else "cpu"
    result = run_experiment("A7_without_feedback", config_without, num_runs=3)
    results.append(result)

    return results


def experiment_A8_nerve_ring(dataset="cifar10"):
    """A8: Compare with and without nerve ring"""
    results = []

    # With nerve ring (baseline)
    config_with = get_baseline_config()
    config_with.dataset = dataset
    config_with.device = "cuda" if torch.cuda.is_available() else "cpu"
    result = run_experiment("A8_with_nerve_ring", config_with, num_runs=3)
    results.append(result)

    # Without nerve ring
    config_without = get_ablation_no_nerve_ring()
    config_without.dataset = dataset
    config_without.device = "cuda" if torch.cuda.is_available() else "cpu"
    result = run_experiment("A8_without_nerve_ring", config_without, num_runs=3)
    results.append(result)

    return results


def experiment_A9_heterogeneous(dataset="cifar10"):
    """A9: Compare homogeneous vs heterogeneous modules"""
    results = []

    # Homogeneous (all CNNs)
    config_homo = get_baseline_config()
    config_homo.dataset = dataset
    config_homo.homogeneous = True
    config_homo.device = "cuda" if torch.cuda.is_available() else "cpu"
    result = run_experiment("A9_homogeneous", config_homo, num_runs=3)
    results.append(result)

    # Heterogeneous (CNN + Transformer + LSTM)
    config_hetero = OctopusNetConfig(
        dataset=dataset,
        homogeneous=False,
        epochs=50,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    result = run_experiment("A9_heterogeneous", config_hetero, num_runs=3)
    results.append(result)

    return results


def experiment_A11_adaptive_threshold(dataset="cifar10"):
    """
    A11: Compare fixed threshold vs adaptive threshold for Forward-Forward.

    Adaptive threshold automatically adjusts based on the midpoint between
    positive and negative goodness values each batch.
    """
    results = []

    # A11a: Fixed threshold (baseline)
    config_fixed = OctopusNetConfig(
        dataset=dataset,
        ff_threshold=2.0,
        ff_adaptive_threshold=False,
        epochs=50,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    result = run_experiment("A11a_fixed_threshold", config_fixed, num_runs=3)
    results.append(result)

    # A11b: Adaptive threshold
    config_adaptive = OctopusNetConfig(
        dataset=dataset,
        ff_threshold=2.0,  # Initial value, will adapt
        ff_adaptive_threshold=True,
        epochs=50,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    result = run_experiment("A11b_adaptive_threshold", config_adaptive, num_runs=3)
    results.append(result)

    return results


def experiment_A10_competition_mechanism(dataset="cifar10"):
    """
    A10: Compare different competition mechanisms (GWT-inspired)

    Tests three competition strategies for the coordinator:
    - A10a: Soft attention (standard softmax) - default
    - A10b: Gumbel-softmax (hard selection, more faithful to GWT)
    - A10c: Top-K sparse attention (only K modules contribute)
    """
    results = []

    # A10a: Soft attention (baseline)
    config_soft = get_baseline_config()
    config_soft.dataset = dataset
    config_soft.competition_type = "soft"
    config_soft.device = "cuda" if torch.cuda.is_available() else "cpu"
    result = run_experiment("A10a_soft_attention", config_soft, num_runs=3)
    results.append(result)

    # A10b: Gumbel-softmax (hard selection)
    config_gumbel = get_gumbel_competition_config()
    config_gumbel.dataset = dataset
    config_gumbel.device = "cuda" if torch.cuda.is_available() else "cpu"
    result = run_experiment("A10b_gumbel_hard", config_gumbel, num_runs=3)
    results.append(result)

    # A10c: Top-2 sparse attention
    config_topk = get_topk_competition_config(k=2)
    config_topk.dataset = dataset
    config_topk.device = "cuda" if torch.cuda.is_available() else "cpu"
    result = run_experiment("A10c_topk_2", config_topk, num_runs=3)
    results.append(result)

    # Additional: Top-1 (winner-take-all)
    config_top1 = get_topk_competition_config(k=1)
    config_top1.dataset = dataset
    config_top1.device = "cuda" if torch.cuda.is_available() else "cpu"
    result = run_experiment("A10d_topk_1_winner_take_all", config_top1, num_runs=3)
    results.append(result)

    return results


def experiment_A12_multiscale(dataset="cifar10"):
    """
    A12: Multi-scale input hierarchy (V1/V2/V4/IT cortex analog).

    A12a: Adaptive kernels only (no resolution change) — control
    A12b: Adaptive kernels + multi-scale input — full bio-inspired hierarchy
    """
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Baseline: original homogeneous config
    config_base = get_baseline_config()
    config_base.dataset = dataset
    config_base.device = device
    results.append(run_experiment("A12_baseline", config_base, num_runs=3))

    # A12a: kernels adapted per resolution but same input size
    config_a = get_adaptive_kernels_only_config(dataset)
    config_a.device = device
    config_a.epochs = 50
    results.append(run_experiment("A12a_adaptive_kernels_only", config_a, num_runs=3))

    # A12b: full multi-scale (different resolution per module)
    config_b = get_multiscale_config(dataset)
    config_b.device = device
    config_b.epochs = 50
    results.append(run_experiment("A12b_multiscale", config_b, num_runs=3))

    return results


def experiment_A15_channel_grouping(dataset="cifar10"):
    """
    A15: Channel grouping FF vs standard FF.
    Single forward pass — channels split into J groups (one per class).
    Goodness computed per group; no x_neg generation needed.
    Based on: Ortiz Torres et al. (arXiv:2504.21662, 2025).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config_std = get_baseline_config()
    config_std.dataset = dataset
    config_std.device = device
    config_std.epochs = 50
    results = [run_experiment("A15_standard_ff", config_std, num_runs=3)]

    config_cg = get_channel_grouping_config(dataset)
    config_cg.device = device
    config_cg.epochs = 50
    results.append(run_experiment("A15_channel_grouping", config_cg, num_runs=3))

    return results


def run_all_experiments(datasets=["mnist", "cifar10"]):
    """
    Run all experiments for the paper.
    """
    all_results = {}

    for dataset in datasets:
        print(f"\n{'*'*70}")
        print(f"* Dataset: {dataset}")
        print(f"{'*'*70}")

        all_results[dataset] = {
            'A1_num_modules': experiment_A1_num_modules(dataset),
            'A2_bottleneck': experiment_A2_bottleneck_size(dataset),
            'A6_resilience': experiment_A6_resilience(dataset),
            'A7_feedback': experiment_A7_feedback(dataset),
            'A8_nerve_ring': experiment_A8_nerve_ring(dataset),
            'A9_heterogeneous': experiment_A9_heterogeneous(dataset),
            'A10_competition': experiment_A10_competition_mechanism(dataset),
            'A11_adaptive_threshold': experiment_A11_adaptive_threshold(dataset),
            'A12_multiscale': experiment_A12_multiscale(dataset),
            'A15_channel_grouping': experiment_A15_channel_grouping(dataset),
        }

    # Save all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'all_experiments_{timestamp}.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


if __name__ == "__main__":
    # Run a quick test with MNIST
    results = run_all_experiments(datasets=["mnist"])
    print("\nAll experiments completed!")

#!/usr/bin/env python3
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.benchmarks import get_supported_benchmarks
from src.estimator import ValidatedEstimator
from src.pinn import PINNNetwork, PINNSolver, Trainer, get_domain_samplers
from experiments.common import (
    collect_run_metadata,
    ensure_dir,
    load_benchmark_configs,
    save_json,
    serialise_training_config,
    validate_benchmark_results,
)


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "validated_benchmarks.json")


def set_reproducibility(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def build_benchmark_configs():
    return load_benchmark_configs(CONFIG_PATH)


def run_single_experiment(benchmark, config, device: str, seed: int):
    interior_sampler, boundary_sampler = get_domain_samplers(
        benchmark.domain,
        config.get("sampler"),
    )
    network = PINNNetwork(**config["network"])
    solver = PINNSolver(
        network=network,
        diffusion_coeff=benchmark.diffusion_coeff,
        source_term=benchmark.source_term,
        boundary_condition=benchmark.boundary_condition,
        domain_sampler=interior_sampler,
        boundary_sampler=boundary_sampler,
        device=device,
    )

    trainer = Trainer(solver, config["training"])
    history = trainer.train()

    estimator = ValidatedEstimator(
        solver=solver,
        benchmark=benchmark,
        fem_mesh_size=max(config["mesh_size"], 96 if benchmark.domain == "unit_square" else 80),
        max_mesh_size=384,
        eval_seed=seed,
    )
    result = estimator.evaluate(training_loss=history.total_loss[-1])
    assert result.effectivity >= 1.0, "Estimator is NOT reliable (η < 1)"

    return {
        "benchmark": benchmark.name,
        "key": benchmark.key,
        "description": benchmark.description,
        "architecture": config["network"],
        "training_config": serialise_training_config(config["training"]),
        "history": {
            "epochs": history.epochs[::100],
            "total_loss": history.total_loss[::100],
            "residual_loss": history.residual_loss[::100],
            "boundary_loss": history.boundary_loss[::100],
        },
        "metrics": result.to_dict(),
    }


def save_results(results, path: str) -> None:
    save_json(results, path)


def _metric_series(results, key):
    return [item["metrics"][key] for item in results]


def _labels(results):
    return [item["benchmark"] for item in results]


def plot_effectivity(results, path: str) -> None:
    labels = _labels(results)
    values = _metric_series(results, "effectivity")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(np.arange(len(labels)), values, color=["#1f77b4", "#2ca02c", "#ff7f0e"])
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.axhline(2.0, color="red", linestyle=":", linewidth=1.0)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.02, f"{val:.2f}", ha="center")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel("Effectivity index η")
    ax.set_title("Validated effectivity indices")
    fig.savefig(path)
    plt.close(fig)


def plot_loss_vs_true_error(results, path: str) -> None:
    losses = _metric_series(results, "training_loss")
    true_errors = _metric_series(results, "true_error_energy")
    labels = _labels(results)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(losses, true_errors, s=90, c="#1f77b4", edgecolors="black")
    for x, y, label in zip(losses, true_errors, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 5), fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training loss")
    ax.set_ylabel("True energy error")
    ax.set_title("Training loss vs true error")
    fig.savefig(path)
    plt.close(fig)


def plot_error_decomposition(results, path: str) -> None:
    labels = _labels(results)
    residual = _metric_series(results, "residual_contribution")
    boundary = _metric_series(results, "boundary_lifting_norm")
    true_error = _metric_series(results, "true_error_energy")
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, residual, label="Residual contribution", color="#1f77b4")
    ax.bar(x, boundary, bottom=residual, label="Boundary lifting", color="#ffbf00")
    ax.plot(x, true_error, "ko--", label="True energy error")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Error magnitude")
    ax.set_title("Estimator decomposition")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def plot_residual_vs_total_error(results, path: str) -> None:
    residual = _metric_series(results, "residual_contribution")
    estimated = _metric_series(results, "estimated_error_energy")
    labels = _labels(results)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(residual, estimated, s=90, c="#2ca02c", edgecolors="black")
    for x, y, label in zip(residual, estimated, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 5), fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Residual contribution")
    ax.set_ylabel("Total estimated error")
    ax.set_title("Residual contribution vs total estimate")
    fig.savefig(path)
    plt.close(fig)


def generate_figures(results, figure_dir: str) -> None:
    os.makedirs(figure_dir, exist_ok=True)
    plot_effectivity(results, os.path.join(figure_dir, "effectivity_index.png"))
    plot_loss_vs_true_error(results, os.path.join(figure_dir, "loss_vs_true_error.png"))
    plot_error_decomposition(results, os.path.join(figure_dir, "error_decomposition.png"))
    plot_residual_vs_total_error(results, os.path.join(figure_dir, "residual_vs_total_error.png"))


def parse_args():
    parser = argparse.ArgumentParser(description="Run the validated PINN error estimation pipeline.")
    parser.add_argument(
        "--benchmark",
        default="all",
        choices=["all", "poisson", "variable_coefficient", "l_shaped"],
        help="Run a single benchmark or the full supported suite.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--device", default="cpu", help="Torch device.")
    parser.add_argument(
        "--results-file",
        default=os.path.join(PROJECT_ROOT, "results", "validated_results.json"),
        help="Path to the JSON results file.",
    )
    parser.add_argument(
        "--metadata-file",
        default=os.path.join(PROJECT_ROOT, "results", "validated_results.meta.json"),
        help="Path to the reproducibility metadata JSON file.",
    )
    parser.add_argument(
        "--figure-dir",
        default=os.path.join(PROJECT_ROOT, "results", "figures"),
        help="Directory for generated figures.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_reproducibility(args.seed)

    benchmarks = get_supported_benchmarks()
    configs = build_benchmark_configs()

    if args.benchmark == "all":
        selected_keys = list(benchmarks.keys())
    else:
        selected_keys = [args.benchmark]

    ensure_dir(os.path.dirname(args.results_file))
    ensure_dir(os.path.dirname(args.metadata_file))

    results = []
    for key in selected_keys:
        benchmark = benchmarks[key]
        print("\n" + "=" * 72)
        print(f"Running benchmark: {benchmark.name}")
        print("=" * 72)
        output = run_single_experiment(benchmark, configs[key], args.device, args.seed + len(results))
        if output["metrics"]["effectivity"] > 2.0:
            print(
                f"WARNING: effectivity is larger than desired for {benchmark.name}: "
                f"{output['metrics']['effectivity']:.4f}"
            )
        results.append(output)

    validate_benchmark_results(results)
    save_results(results, args.results_file)
    metadata = collect_run_metadata(
        project_root=PROJECT_ROOT,
        entrypoint="experiments/run.py",
        seed=args.seed,
        device=args.device,
        config_path=CONFIG_PATH,
        extra={"benchmarks": selected_keys},
    )
    save_json(metadata, args.metadata_file)
    generate_figures(results, args.figure_dir)

    print("\nSummary")
    print("-" * 72)
    for item in results:
        metrics = item["metrics"]
        print(
            f"{item['benchmark']:<32} "
            f"true={metrics['true_error_energy']:.4e} "
            f"est={metrics['estimated_error_energy']:.4e} "
            f"eta={metrics['effectivity']:.4f}"
        )
    print(f"\nResults saved to {args.results_file}")
    print(f"Metadata saved to {args.metadata_file}")
    print(f"Figures saved to {args.figure_dir}")


if __name__ == "__main__":
    main()

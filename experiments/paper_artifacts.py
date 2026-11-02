import argparse
import json
import math
import os
import sys
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.common import (  # noqa: E402
    collect_run_metadata,
    ensure_dir,
    save_json,
    validate_benchmark_results,
    validate_paper_artifact_payload,
)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "figure.dpi": 180,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.frameon": True,
        # Embed TrueType fonts (Type 42) so that the PDFs satisfy publisher font checks.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "validated_benchmarks.json")
SQRT2 = math.sqrt(2.0)


# ---------------------------------------------------------------------------------------
# Loading and derived quantities
# ---------------------------------------------------------------------------------------
def load_results(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "results" in payload:
        payload = payload["results"]
    validate_benchmark_results(payload)
    for item in payload:
        for key in ("estimator_pythagoras", "estimator_variable", "effectivity_pythagoras",
                    "effectivity_variable", "boundary_lifting_a_weighted", "mesh_size_used"):
            if key not in item["metrics"]:
                raise ValueError(
                    f"{item['benchmark']}: metrics lack '{key}'. Regenerate the results with "
                    "the current experiments/run.py before building paper artifacts."
                )
    return payload


def _is_variable(metrics: Dict) -> bool:
    """True for the variable-coefficient benchmark (coefficient upper bound beta != 1)."""
    return abs(float(metrics.get("coefficient_upper_bound", 1.0)) - 1.0) > 1e-12


def build_ablation(results: List[Dict]) -> List[Dict]:
    ablation = []
    for item in results:
        m = item["metrics"]
        true_error = m["true_error_energy"]
        residual_only = m["residual_contribution"]
        lifting_only = m["boundary_lifting_norm"]
        full_estimate = m["estimated_error_energy"]
        ablation.append(
            {
                "benchmark": item["benchmark"],
                "key": item["key"],
                "true_error_energy": true_error,
                "residual_only": residual_only,
                "lifting_only": lifting_only,
                "full_estimator": full_estimate,
                "eta_residual_only": residual_only / true_error,
                "eta_lifting_only": lifting_only / true_error,
                "eta_full": full_estimate / true_error,
            }
        )
    return ablation


def build_convergence(results: List[Dict]) -> List[Dict]:
    """Mesh-ladder entries read from the stored ``mesh_history`` (one fixed network).

    The exact error of a fixed network does not depend on the estimator mesh, so it is the
    same value at every level.
    """
    convergence = []
    for item in results:
        m = item["metrics"]
        true_error = m["true_error_energy"]
        entries = []
        for level in m["mesh_history"]:
            entries.append(
                {
                    "mesh_size": float(level["mesh_size"]),
                    "true_error_energy": true_error,
                    "true_error_l2": m["true_error_l2"],
                    "estimated_error_energy": level["estimate"],
                    "effectivity": level["estimate"] / true_error,
                    "residual_contribution": level["residual_contribution"],
                    "boundary_contribution": level["boundary_contribution"],
                    "estimate_pythagoras": level["estimate_pythagoras"],
                    "effectivity_pythagoras": level["estimate_pythagoras"] / true_error,
                    "estimate_variable": level["estimate_variable"],
                    "effectivity_variable": level["estimate_variable"] / true_error,
                    "training_loss": m["training_loss"],
                }
            )
        convergence.append(
            {
                "benchmark": item["benchmark"],
                "key": item["key"],
                "mesh_ladder": [entry["mesh_size"] for entry in entries],
                "entries": entries,
            }
        )
    return convergence


# ---------------------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------------------
def _latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def _write_lines(lines: List[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines + [""]))


def write_benchmark_table(results: List[Dict], path: str) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Benchmark & Mesh & $\|\nabla e\|$ & $\alpha^{-1}\|R\|_{H^{-1}}$ & $\|\nabla w\|$ & $\|w\|_a$ "
        r"& $\eta_\perp/\|\nabla e\|$ & $\eta_\oplus/\|\nabla e\|$ & $\eta_a/\|\nabla e\|$ \\",
        r"\midrule",
    ]
    for item in results:
        m = item["metrics"]
        variable = _is_variable(m)
        w_a = f"{m['boundary_lifting_a_weighted']:.4e}" if variable else "--"
        eta_a = f"{m['effectivity_variable']:.4f}" if variable else "--"
        lines.append(
            f"{_latex_escape(item['benchmark'])} & {int(m['mesh_size_used'])} & "
            f"{m['true_error_energy']:.4e} & {m['residual_contribution']:.4e} & "
            f"{m['boundary_lifting_norm']:.4e} & {w_a} & "
            f"{m['effectivity_pythagoras']:.4f} & {m['effectivity']:.4f} & {eta_a} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write_lines(lines, path)


def write_ablation_table(ablation: List[Dict], path: str) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Benchmark & $\|\nabla e\|$ & $\alpha^{-1}\|R\|_{H^{-1}}$ & $\|\nabla w\|$ "
        r"& $\eta_r$ & $\eta_\lambda$ & $\eta_\oplus/\|\nabla e\|$ \\",
        r"\midrule",
    ]
    for item in ablation:
        lines.append(
            f"{_latex_escape(item['benchmark'])} & {item['true_error_energy']:.4e} & "
            f"{item['residual_only']:.4e} & {item['lifting_only']:.4e} & "
            f"{item['eta_residual_only']:.4f} & {item['eta_lifting_only']:.4f} & "
            f"{item['eta_full']:.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write_lines(lines, path)


def write_loss_table(results: List[Dict], path: str) -> None:
    # Only the final total loss is tabulated. The last entries of history["residual_loss"] and
    # history["boundary_loss"] do not match the loss printed at the end of training for the
    # Poisson and L-shaped runs, so they are deliberately not reported.
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Benchmark & Final training loss & $\|\nabla e\|$ \\",
        r"\midrule",
    ]
    for item in results:
        m = item["metrics"]
        lines.append(
            f"{_latex_escape(item['benchmark'])} & {m['training_loss']:.4e} & "
            f"{m['true_error_energy']:.4e} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write_lines(lines, path)


# ---------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------
def _save_figure(fig, stem: str) -> None:
    fig.savefig(stem + ".png")
    fig.savefig(stem + ".pdf")
    plt.close(fig)


def plot_benchmark_effectivity(results: List[Dict], stem: str) -> None:
    labels = [item["benchmark"] for item in results]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(x - 0.08, [r["metrics"]["effectivity_pythagoras"] for r in results], "o",
            color="#1f77b4", label=r"$\eta_\perp/\|\nabla e\|$")
    ax.plot(x + 0.08, [r["metrics"]["effectivity"] for r in results], "s",
            color="#2ca02c", label=r"$\eta_\oplus/\|\nabla e\|$")
    variable = [i for i, r in enumerate(results) if _is_variable(r["metrics"])]
    if variable:
        ax.plot([i + 0.24 for i in variable],
                [results[i]["metrics"]["effectivity_variable"] for i in variable], "^",
                color="#ff7f0e", label=r"$\eta_a/\|\nabla e\|$")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.axhline(SQRT2, color="gray", linestyle=":", linewidth=1.0, label=r"$\sqrt{2}$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel("Effectivity index")
    ax.legend()
    _save_figure(fig, stem)


def plot_ablation(ablation: List[Dict], stem: str) -> None:
    labels = [item["benchmark"] for item in ablation]
    x = np.arange(len(labels))
    width = 0.22
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width, [i["residual_only"] for i in ablation], width=width,
           label=r"Residual term $\alpha^{-1}\|R\|_{H^{-1}}$", color="#1f77b4")
    ax.bar(x, [i["lifting_only"] for i in ablation], width=width,
           label=r"Lifting term $\|\nabla w\|$", color="#ffbf00")
    ax.bar(x + width, [i["full_estimator"] for i in ablation], width=width,
           label=r"Sum $\eta_\oplus$", color="#2ca02c")
    ax.plot(x, [i["true_error_energy"] for i in ablation], "ko--", label=r"Exact error $\|\nabla e\|$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Magnitude")
    ax.legend()
    _save_figure(fig, stem)


def plot_convergence_errors(convergence: List[Dict], stem: str) -> None:
    fig, axes = plt.subplots(1, len(convergence), figsize=(4.8 * len(convergence), 4.2), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, bench in zip(axes, convergence):
        mesh = [e["mesh_size"] for e in bench["entries"]]
        ax.plot(mesh, [e["true_error_energy"] for e in bench["entries"]], "k--", linewidth=1.4,
                label=r"Exact error $\|\nabla e\|$")
        ax.plot(mesh, [e["estimate_pythagoras"] for e in bench["entries"]], "o-", color="#1f77b4",
                label=r"$\eta_\perp$")
        ax.plot(mesh, [e["estimated_error_energy"] for e in bench["entries"]], "s-", color="#2ca02c",
                label=r"$\eta_\oplus$")
        ax.set_xscale("log", base=2)
        # Log scale: a linear axis zoomed to the data would exaggerate differences of a few percent.
        ax.set_yscale("log")
        ax.set_xlabel("Estimator mesh (intervals per side)")
        ax.set_title(bench["benchmark"])
    axes[0].set_ylabel("Energy-seminorm error")
    axes[-1].legend()
    _save_figure(fig, stem)


def plot_convergence_effectivity(convergence: List[Dict], stem: str) -> None:
    fig, axes = plt.subplots(1, len(convergence), figsize=(4.8 * len(convergence), 4.2), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, bench in zip(axes, convergence):
        mesh = [e["mesh_size"] for e in bench["entries"]]
        ax.plot(mesh, [e["effectivity_pythagoras"] for e in bench["entries"]], "o-", color="#1f77b4",
                label=r"$\eta_\perp/\|\nabla e\|$")
        ax.plot(mesh, [e["effectivity"] for e in bench["entries"]], "s-", color="#2ca02c",
                label=r"$\eta_\oplus/\|\nabla e\|$")
        if bench["key"] == "variable_coefficient":
            ax.plot(mesh, [e["effectivity_variable"] for e in bench["entries"]], "^-", color="#ff7f0e",
                    label=r"$\eta_a/\|\nabla e\|$")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        ax.axhline(SQRT2, color="gray", linestyle=":", linewidth=1.0)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Estimator mesh (intervals per side)")
        ax.set_title(bench["benchmark"])
    axes[0].set_ylabel("Effectivity index")
    # One legend for all panels (eta_a appears only in the variable-coefficient panel).
    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    axes[0].legend(handles, labels)
    _save_figure(fig, stem)


# ---------------------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------------------
def write_summary(results: List[Dict], ablation: List[Dict], convergence: List[Dict],
                  source: str, path: str) -> None:
    rows = []
    for item in results:
        m = item["metrics"]
        ratio = m["residual_contribution"] / m["boundary_lifting_norm"]
        rows.append(
            f"| {item['benchmark']} | {int(m['mesh_size_used'])} | {m['true_error_energy']:.4e} | "
            f"{m['effectivity_pythagoras']:.4f} | {m['effectivity']:.4f} | {ratio:.4f} | "
            f"{m['training_loss']:.4e} |"
        )
    ladders = []
    for bench in convergence:
        levels = ", ".join(
            f"{int(e['mesh_size'])}: {e['effectivity_pythagoras']:.4f}" for e in bench["entries"]
        )
        ladders.append(f"* {bench['benchmark']} (mesh: eta_perp / exact error): {levels}")

    summary = f"""# Experiment Summary

Generated by `experiments/paper_artifacts.py` from `{os.path.basename(source)}`. Do not edit by hand.

## Results on the finest mesh level (one trained network per benchmark)

| Benchmark | Mesh | Exact error | eta_perp / error | eta_sum / error | residual term / lifting term | Training loss |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

`eta_perp` is the Pythagorean combination and `eta_sum` the sum of the residual term and the
lifting term. For a constant coefficient the continuous value of `eta_perp / error` is 1 and
that of `eta_sum / error` lies in [1, sqrt(2)].

## Mesh ladder

{chr(10).join(ladders)}

## Reading these numbers

* The values are discretization estimates of the exact norms, not certified bounds.
* One random seed and one trained network per benchmark; no statement about other seeds.
* A difference between a computed index and its continuous value contains the discretization
  error of the dual norm, of the lifting, and of the reference error. The mesh ladder above
  shows how it changes under refinement.
* No bound is proved for the sum form with a variable coefficient; `eta_a` is a numerical
  diagnostic.
"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(summary)


# ---------------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Generate paper tables and figures from the stored results.")
    parser.add_argument("--results-file", default=os.path.join(PROJECT_ROOT, "results", "validated_results.json"))
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "results", "paper"))
    parser.add_argument("--figure-dir", default=os.path.join(PROJECT_ROOT, "paper", "figures"))
    parser.add_argument("--tables-dir", default=os.path.join(PROJECT_ROOT, "paper", "tables"))
    parser.add_argument("--summary-file", default=os.path.join(PROJECT_ROOT, "docs", "EXPERIMENT_SUMMARY.md"))
    parser.add_argument("--metadata-file", default=os.path.join(PROJECT_ROOT, "results", "paper", "artifact_metadata.json"))
    parser.add_argument("--device", default="cpu", help="Recorded in the metadata only (nothing is trained here).")
    parser.add_argument("--seed", type=int, default=42, help="Recorded in the metadata only (nothing is trained here).")
    return parser.parse_args()


def main():
    args = parse_args()
    for directory in (args.results_dir, args.figure_dir, args.tables_dir,
                      os.path.dirname(args.metadata_file), os.path.dirname(args.summary_file)):
        ensure_dir(directory)

    results = load_results(args.results_file)
    ablation = build_ablation(results)
    convergence = build_convergence(results)

    metadata = collect_run_metadata(
        project_root=PROJECT_ROOT,
        entrypoint="experiments/paper_artifacts.py",
        seed=args.seed,
        device=args.device,
        config_path=CONFIG_PATH,
        extra={"source_results": os.path.abspath(args.results_file)},
    )
    payload = {
        "metadata": metadata,
        "source_results": os.path.abspath(args.results_file),
        "benchmarks": results,
        "ablation": ablation,
        "convergence": convergence,
    }
    validate_paper_artifact_payload(payload)
    save_json(payload, os.path.join(args.results_dir, "paper_artifacts.json"))
    save_json({"metadata": metadata, "ablation": ablation}, os.path.join(args.results_dir, "ablation_results.json"))
    save_json({"metadata": metadata, "convergence": convergence}, os.path.join(args.results_dir, "convergence_results.json"))
    save_json(metadata, args.metadata_file)

    write_benchmark_table(results, os.path.join(args.tables_dir, "benchmark_results.tex"))
    write_ablation_table(ablation, os.path.join(args.tables_dir, "ablation_results.tex"))
    write_loss_table(results, os.path.join(args.tables_dir, "loss_vs_error.tex"))

    plot_benchmark_effectivity(results, os.path.join(args.figure_dir, "effectivity_benchmarks"))
    plot_ablation(ablation, os.path.join(args.figure_dir, "ablation_study"))
    plot_convergence_errors(convergence, os.path.join(args.figure_dir, "convergence_errors"))
    plot_convergence_effectivity(convergence, os.path.join(args.figure_dir, "convergence_effectivity"))

    write_summary(results, ablation, convergence, args.results_file, args.summary_file)

    print(f"Paper artifact JSON saved to {args.results_dir}")
    print(f"Artifact metadata saved to {args.metadata_file}")
    print(f"Paper figures saved to {args.figure_dir}")
    print(f"LaTeX tables saved to {args.tables_dir}")
    print(f"Experiment summary saved to {args.summary_file}")


if __name__ == "__main__":
    main()
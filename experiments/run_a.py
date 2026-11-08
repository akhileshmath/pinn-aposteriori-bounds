#!/usr/bin/env python3
"""Multi-seed experiment driver for the a-energy (Theorem A) study.

One run = (benchmark, variant, seed).  Each run trains a PINN (float64 by default), saves its
weights, evaluates the a-energy estimator on a nested mesh ladder against an accurate reference
error, records boundary diagnostics and fixed-sample losses, and writes ONE json file.  Runs that
already have a json file are skipped, so the script can be stopped and restarted at any time.

Examples (PowerShell, one line each):
    python experiments/run_a.py --quick --out results/a_quick                       # pipeline check, minutes
    python experiments/run_a.py --seeds 0 1 2 --jobs 2 --out results/a_energy_runs
    python experiments/run_a.py --benchmarks l_shaped --variants baseline corner_refined --seeds 0 1 2 3 4
    python experiments/run_a.py --summarize --out results/a_energy_runs              # table from existing runs
"""
import argparse
import concurrent.futures as cf
import csv
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BENCH_CONFIG = os.path.join(PROJECT_ROOT, "configs", "validated_benchmarks.json")
VARIANT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "a_energy_variants.json")
DEFAULT_LADDER = {"unit_square": [32, 64, 128], "l_shaped": [64, 128, 256, 512]}
QUICK_LADDER = {"unit_square": [16, 32], "l_shaped": [16, 32]}


# ---------------------------------------------------------------------------------- configuration
def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _merge(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = json.loads(json.dumps(base))
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def build_run_config(benchmark_key: str, variant: str, quick: bool) -> Dict[str, Any]:
    bench = _load_json(BENCH_CONFIG)[benchmark_key]
    variants = _load_json(VARIANT_CONFIG)
    spec = variants[variant]
    if benchmark_key not in spec["applies_to"]:
        raise ValueError(f"variant '{variant}' does not apply to '{benchmark_key}'")
    base_training = dict(bench["training"])
    base_network = dict(bench["network"])
    if quick:  # pipeline check only; NOT for reporting.  Applied to the BASE so that a variant's own
        # overrides (e.g. n_boundary for the sparse-boundary variants) still win over the quick defaults.
        base_training.update(adam_epochs=150, lbfgs_epochs=3, n_collocation=300, n_boundary=100,
                             resample_every=50, print_every=10**9)
        base_network["hidden_dims"] = [16, 16]
    return {
        "network": _merge(base_network, spec.get("network")),
        "training": _merge(base_training, spec.get("training")),
        "sampler": _merge(bench.get("sampler", {}), spec.get("sampler")),
    }


def run_id(benchmark: str, variant: str, seed: int) -> str:
    return f"{benchmark}__{variant}__seed{seed}"


def enumerate_runs(benchmarks: List[str], variants: List[str], seeds: List[int]) -> List[Tuple[str, str, int]]:
    spec = _load_json(VARIANT_CONFIG)
    runs = []
    for seed in seeds:
        for b in benchmarks:
            for v in variants:
                if b in spec[v]["applies_to"]:
                    runs.append((b, v, seed))
    return runs


# ---------------------------------------------------------------------------------- one run
def _thin(values, step: int = 100):
    values = list(values)
    thinned = values[::step]
    if values and (len(values) - 1) % step != 0:
        thinned.append(values[-1])
    return thinned


def run_one(task: Dict[str, Any]) -> Dict[str, Any]:
    """Train + evaluate one (benchmark, variant, seed).  Returns a small status dict."""
    import torch

    from experiments.common import _git_commit
    from src.a_energy import observed_order_and_limit
    from src.a_energy.torch_bridge import evaluate_for_solver
    from src.benchmarks import get_supported_benchmarks
    from src.pinn import TrainingConfig, fixed_sample_losses, train_pinn

    b, v, seed, out_dir = task["benchmark"], task["variant"], task["seed"], task["out"]
    rid = run_id(b, v, seed)
    path = os.path.join(out_dir, "runs", rid + ".json")
    if os.path.exists(path) and not task["force"]:
        return {"id": rid, "status": "skipped"}
    if task.get("threads"):
        torch.set_num_threads(int(task["threads"]))

    try:
        cfg = build_run_config(b, v, task["quick"])
        benchmark = get_supported_benchmarks()[b]
        ladder = task["ladder"] or (QUICK_LADDER if task["quick"] else DEFAULT_LADDER)[benchmark.domain]
        t0 = time.time()
        solver, history = train_pinn(
            benchmark, cfg["network"], TrainingConfig(**cfg["training"]), cfg["sampler"],
            seed=seed, device="cpu", dtype=task["dtype"],
        )
        train_seconds = time.time() - t0

        os.makedirs(os.path.join(out_dir, "weights"), exist_ok=True)
        torch.save(solver.net.state_dict(), os.path.join(out_dir, "weights", rid + ".pt"))

        fixed = fixed_sample_losses(solver, benchmark)
        t1 = time.time()
        ev = evaluate_for_solver(solver, benchmark, ladder=ladder, ref_cells=32 if task["quick"] else 128)
        eval_seconds = time.time() - t1
        etas = [lv["eta"] for lv in ev["levels"]]
        extrap = observed_order_and_limit(etas, assumed_order=4.0 / 3.0 if benchmark.domain == "l_shaped" else 2.0)
        true_a = ev["true_error_a"]
        for key in ("limit", "limit_assumed"):
            extrap[key + "_over_true"] = None if extrap[key] is None else extrap[key] / true_a

        record = {
            "id": rid, "benchmark": b, "variant": v, "seed": seed, "quick": task["quick"],
            "dtype": task["dtype"], "config": cfg, "ladder": list(ladder),
            "train_seconds": train_seconds, "eval_seconds": eval_seconds,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(PROJECT_ROOT), "torch": torch.__version__, "numpy": np.__version__,
            "history": {
                "epochs": _thin(history.epochs), "total_loss": _thin(history.total_loss),
                "residual_loss": _thin(history.residual_loss), "boundary_loss": _thin(history.boundary_loss),
                "tangential_loss": _thin(history.tangential_loss),
            },
            "fixed_sample": fixed,
            "evaluation": ev,
            "extrapolation": extrap,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, default=float)
        os.replace(tmp, path)
        return {"id": rid, "status": "done", "train_s": round(train_seconds, 1),
                "eta_over_true": ev["levels"][-1]["eta_over_true"]}
    except Exception:  # keep going; the failure is recorded next to the results
        os.makedirs(os.path.join(out_dir, "errors"), exist_ok=True)
        with open(os.path.join(out_dir, "errors", rid + ".txt"), "w", encoding="utf-8") as handle:
            handle.write(traceback.format_exc())
        return {"id": rid, "status": "failed"}


# ---------------------------------------------------------------------------------- summary
def load_runs(out_dir: str) -> List[Dict[str, Any]]:
    folder = os.path.join(out_dir, "runs")
    if not os.path.isdir(folder):
        return []
    runs = []
    for name in sorted(os.listdir(folder)):
        if name.endswith(".json"):
            runs.append(_load_json(os.path.join(folder, name)))
    return runs


def _row(r: Dict[str, Any]) -> Dict[str, float]:
    ev, fs = r["evaluation"], r["fixed_sample"]
    last = ev["levels"][-1]
    return {
        "true_error_a": ev["true_error_a"], "eta": last["eta"], "eta_over_true": last["eta_over_true"],
        "dual": last["dual_norm"], "lifting": last["lifting_norm"],
        "bnd_l2": ev["boundary"]["l2"], "bnd_tan": ev["boundary"]["tangential_h1"],
        "fixed_res_loss": fs["residual_loss"], "fixed_bnd_loss": fs["boundary_loss"],
        "extrap_over_true": r["extrapolation"].get("limit_over_true"),
        # energy of the lifting per unit L2 boundary mismatch: large => the mismatch is oscillatory
        "lift_per_l2": last["lifting_norm"] / ev["boundary"]["l2"] if ev["boundary"]["l2"] > 1e-12 else float("nan"),
        "train_s": r["train_seconds"],
    }


def summarize(out_dir: str) -> None:
    runs = load_runs(out_dir)
    if not runs:
        print(f"No runs found in {out_dir}/runs")
        return
    if any(r.get("quick") for r in runs):
        print("WARNING: results contain --quick runs (tiny networks, few epochs): pipeline check only.\n")
    groups: Dict[Tuple[str, str], List[Dict[str, float]]] = {}
    for r in runs:
        groups.setdefault((r["benchmark"], r["variant"]), []).append(_row(r))
    groups_of_columns = [
        ["true_error_a", "eta_over_true", "extrap_over_true", "lifting", "dual"],
        ["lift_per_l2", "bnd_l2", "bnd_tan", "fixed_res_loss", "fixed_bnd_loss", "train_s"],
    ]
    csv_rows = []
    for cols in groups_of_columns:
        header = f"{'benchmark':<21}{'variant':<15}{'n':>3} " + " ".join(f"{c:>17}" for c in cols)
        print(header)
        print("-" * len(header))
        for (b, v), rows in sorted(groups.items()):
            cells = []
            for c in cols:
                vals = np.array([row[c] if row[c] is not None else np.nan for row in rows], dtype=float)
                sd = np.nanstd(vals, ddof=1) if np.sum(~np.isnan(vals)) > 1 else float("nan")
                mean = np.nanmean(vals) if np.any(~np.isnan(vals)) else float("nan")
                cells.append(f"{mean:.4f}±{sd:.0e}" if c in ("eta_over_true", "extrap_over_true") else f"{mean:.2e}±{sd:.0e}")
                csv_rows.append({"benchmark": b, "variant": v, "metric": c, "n": int(np.sum(~np.isnan(vals))), "mean": mean, "std": sd})
            print(f"{b:<21}{v:<15}{len(rows):>3} " + " ".join(f"{x:>17}" for x in cells))
        print()
    print("lift_per_l2 = lifting energy / ||g-uhat||_L2(Gamma): compare it ACROSS VARIANTS at similar L2 mismatch;")
    print("a tangential penalty that lowers it supports 'the lifting is controlled by the oscillation, not by the size'.")
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["benchmark", "variant", "metric", "n", "mean", "std"])
        writer.writeheader()
        writer.writerows(csv_rows)
    errors = os.path.join(out_dir, "errors")
    if os.path.isdir(errors) and os.listdir(errors):
        print(f"\n{len(os.listdir(errors))} failed run(s): see {errors}")


# ---------------------------------------------------------------------------------- CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmarks", nargs="+", default=["poisson", "variable_coefficient", "l_shaped"])
    ap.add_argument("--variants", nargs="+", default=["baseline", "wb1", "wb100", "tangent_g1", "tangent_g10", "hardbc", "corner_refined"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "results", "a_energy_runs"))
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--ladder", type=int, nargs="+", default=None, help="override mesh ladder (cells per side)")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    ap.add_argument("--quick", action="store_true", help="tiny networks / few epochs: pipeline check only")
    ap.add_argument("--force", action="store_true", help="recompute runs that already have a json")
    ap.add_argument("--summarize", action="store_true", help="only print the summary of existing runs")
    ap.add_argument("--dry-run", action="store_true", help="list the runs and exit")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.summarize:
        summarize(args.out)
        return
    unknown = set(args.variants) - set(k for k in _load_json(VARIANT_CONFIG) if not k.startswith("_"))
    if unknown:
        raise SystemExit(f"unknown variants: {sorted(unknown)}")
    runs = enumerate_runs(args.benchmarks, args.variants, args.seeds)
    todo = [r for r in runs if args.force or not os.path.exists(os.path.join(args.out, "runs", run_id(*r) + ".json"))]
    print(f"{len(runs)} runs requested, {len(runs) - len(todo)} already done, {len(todo)} to do -> {args.out}")
    if args.dry_run:
        for r in todo:
            print("  ", run_id(*r))
        return
    threads = max(1, (os.cpu_count() or 1) // max(args.jobs, 1))
    tasks = [{"benchmark": b, "variant": v, "seed": s, "out": args.out, "dtype": args.dtype, "ladder": args.ladder,
              "quick": args.quick, "force": args.force, "threads": threads} for b, v, s in todo]
    start = time.time()
    if args.jobs <= 1:
        results = []
        for i, t in enumerate(tasks, 1):
            res = run_one(t)
            results.append(res)
            print(f"[{i}/{len(tasks)}] {res}  elapsed {time.time() - start:.0f}s", flush=True)
    else:
        results = []
        with cf.ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as pool:
            futures = [pool.submit(run_one, t) for t in tasks]
            for i, fut in enumerate(cf.as_completed(futures), 1):
                res = fut.result()
                results.append(res)
                print(f"[{i}/{len(tasks)}] {res}  elapsed {time.time() - start:.0f}s", flush=True)
    failed = [r["id"] for r in results if r["status"] == "failed"]
    print(f"\nFinished in {time.time() - start:.0f}s; failed: {failed if failed else 'none'}")
    summarize(args.out)


if __name__ == "__main__":
    main()

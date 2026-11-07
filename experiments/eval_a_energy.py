#!/usr/bin/env python3
"""Evaluate the a-energy estimator (Theorem A) on ALREADY TRAINED networks (no retraining).

    python experiments/eval_a_energy.py --benchmarks all --weights-dir results/weights \
        --ladder 32 64 128 --out results/a_energy/existing_networks.json

Prints, per benchmark, the new float64 weak-form numbers next to the numbers stored by the old
pipeline in results/validated_results.json, so that the unit-square 'floor' and the L-shaped
reference error can be diagnosed directly.
"""
import argparse
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.a_energy.torch_bridge import evaluate_for_solver  # noqa: E402
from src.benchmarks import get_supported_benchmarks  # noqa: E402
from src.pinn import PINNNetwork  # noqa: E402

CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "validated_benchmarks.json")


def _old_metrics(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    data = data.get("results", data) if isinstance(data, dict) else data
    return {item["key"]: item["metrics"] for item in data}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", default=["all"])
    ap.add_argument("--weights-dir", default=os.path.join(PROJECT_ROOT, "results", "weights"))
    ap.add_argument("--ladder", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--ref-cells", type=int, default=128)
    ap.add_argument("--quad-order", type=int, default=4)
    ap.add_argument("--old-results", default=os.path.join(PROJECT_ROOT, "results", "validated_results.json"))
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "results", "a_energy", "existing_networks.json"))
    args = ap.parse_args()

    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        configs = json.load(handle)
    benchmarks = get_supported_benchmarks()
    keys = list(benchmarks) if args.benchmarks == ["all"] else args.benchmarks
    old = _old_metrics(args.old_results)

    out = {}
    for key in keys:
        net = PINNNetwork(**configs[key]["network"])
        net.load_state_dict(torch.load(os.path.join(args.weights_dir, f"{key}.pt"), map_location="cpu"))
        res = evaluate_for_solver(net, benchmarks[key], args.ladder, args.quad_order, args.ref_cells)
        out[key] = res

        print(f"\n=== {key} (alpha={res['alpha']}, beta={res['beta']}) ===")
        print(f"  reference: {res['ref_method']}, self-convergence rel. change {res['ref_self_convergence_rel']:.2e}")
        print(f"  true ||e||_a = {res['true_error_a']:.8e}   true ||grad e|| = {res['true_error_grad']:.8e}")
        if key in old:
            print(f"  [old pipeline] ||grad e|| = {old[key]['true_error_energy']:.8e}")
        print("  n_cells   dual_norm      lifting       eta_a        eta_a/||e||_a")
        for lv in res["levels"]:
            print(f"  {lv['n_cells']:7d}   {lv['dual_norm']:.6e}  {lv['lifting_norm']:.6e}  {lv['eta']:.6e}  {lv['eta_over_true']:.6f}")
        b = res["boundary"]
        print(f"  boundary: ||g-uhat||_L2(Gamma) = {b['l2']:.3e}, ||d_tau(g-uhat)||_L2(Gamma) = {b['tangential_h1']:.3e}")
        lo, hi = res["unweighted_interval_finest"]
        print(f"  unweighted interval for ||grad e||: [{lo:.6e}, {hi:.6e}]")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Root-cause diagnostics on SAVED weights (no retraining).

    python experiments/diagnose_runs.py --runs results/a_energy_runs

For every run in <runs>/runs/*.json it reloads <runs>/weights/<id>.pt (float64) and computes
  * the strong-form residual  ||r||_{L2(Omega)}  by an accurate quadrature, and (L-shape only) the share
    of ||r||^2 that sits within distance 0.1 / 0.3 of the re-entrant corner;
  * the boundary mismatch ||g - uhat||_{L2(edge)}^2 edge by edge, and (L-shape) the share carried by the
    two re-entrant edges touching the corner.
It answers WHERE a variant's extra residual / mismatch lives, which separates 'optimisation competition
near the corner' from 'uniformly worse'.  Output: <runs>/diagnostics.json and a table per variant.
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.run_a import load_runs  # noqa: E402
from src.a_energy.references import _L_EDGES, _SQUARE_EDGES, _edge_rule, lshape_graded_rule  # noqa: E402
from src.a_energy.torch_bridge import make_uhat_eval  # noqa: E402
from src.benchmarks import get_supported_benchmarks  # noqa: E402
from src.pinn import PINNNetwork, PINNSolver, get_domain_samplers  # noqa: E402

EDGE_NAMES_L = ["bottom", "reentrant_x0", "reentrant_y0", "right", "top", "left"]
EDGE_NAMES_SQ = ["bottom", "right", "top", "left"]


def diagnose(record: Dict[str, Any], weights_path: str) -> Dict[str, Any]:
    bm = get_supported_benchmarks()[record["benchmark"]]
    net = PINNNetwork(**record["config"]["network"])
    net.load_state_dict(torch.load(weights_path, map_location="cpu"))
    net = net.double().eval()
    interior, boundary = get_domain_samplers(bm.domain)
    solver = PINNSolver(net, bm.diffusion_coeff, bm.source_term, bm.boundary_condition, interior, boundary, "cpu",
                        domain=bm.domain)
    out: Dict[str, Any] = {"id": record["id"], "benchmark": record["benchmark"], "variant": record["variant"],
                           "seed": record["seed"]}

    # ---- strong-form residual by quadrature (L-shape: graded polar rule resolves the corner)
    if bm.domain == "l_shaped":
        pts, w = lshape_graded_rule(96, 64)
        x = torch.tensor(pts, dtype=torch.float64)
        res2 = solver.compute_pde_residual(x).detach().numpy().reshape(-1) ** 2
        tot = float(np.sum(w * res2))
        rho = np.hypot(pts[:, 0], pts[:, 1])
        out["residual_l2"] = float(np.sqrt(tot))
        for radius in (0.1, 0.3):
            out[f"residual_share_within_{radius}"] = float(np.sum((w * res2)[rho < radius]) / tot)
        out["domain_area_share_within_0.1"] = float(np.sum(w[rho < 0.1]) / np.sum(w))
        out["domain_area_share_within_0.3"] = float(np.sum(w[rho < 0.3]) / np.sum(w))

    else:  # unit square: smooth solution, a tensor Gauss rule is ample
        xg, wg = np.polynomial.legendre.leggauss(64)
        xg, wg = 0.5 * (xg + 1.0), 0.5 * wg
        gx, gy = np.meshgrid(xg, xg, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        w = (wg[:, None] * wg[None, :]).ravel()
        x = torch.tensor(pts, dtype=torch.float64)
        res2 = solver.compute_pde_residual(x).detach().numpy().reshape(-1) ** 2
        out["residual_l2"] = float(np.sqrt(np.sum(w * res2)))

    # ---- boundary mismatch edge by edge
    uhat_eval = make_uhat_eval(net)
    edges, names = (_L_EDGES, EDGE_NAMES_L) if bm.domain == "l_shaped" else (_SQUARE_EDGES, EDGE_NAMES_SQ)
    per_edge = {}
    for (p0, p1, _n, _c), name in zip(edges, names):
        pts, wq, _ = _edge_rule(p0, p1)
        g = bm.boundary_condition(torch.tensor(pts, dtype=torch.float64)).detach().numpy().reshape(-1)
        d = g - uhat_eval(pts)[0]
        per_edge[name] = float(np.sum(wq * d**2))
    total = sum(per_edge.values())
    out["boundary_l2_sq_by_edge"] = per_edge
    out["boundary_l2"] = float(np.sqrt(total))
    if bm.domain == "l_shaped":
        out["mismatch_share_reentrant_edges"] = float((per_edge["reentrant_x0"] + per_edge["reentrant_y0"]) / total)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.path.join(PROJECT_ROOT, "results", "a_energy_runs"))
    ap.add_argument("--benchmarks", nargs="+", default=None)
    args = ap.parse_args(argv)

    results: List[Dict[str, Any]] = []
    for rec in load_runs(args.runs):
        if args.benchmarks and rec["benchmark"] not in args.benchmarks:
            continue
        wp = os.path.join(args.runs, "weights", rec["id"] + ".pt")
        if os.path.exists(wp):
            results.append(diagnose(rec, wp))
    if not results:
        raise SystemExit("no runs with saved weights found")
    with open(os.path.join(args.runs, "diagnostics.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for r in results:
        groups.setdefault((r["benchmark"], r["variant"]), []).append(r)
    print(f"{'benchmark':<21}{'variant':<17}{'n':>3} {'residual_L2':>12} {'res share r<0.1':>16} {'(area share)':>13} "
          f"{'res share r<0.3':>16} {'mismatch share@reentrant':>25}")
    for (b, v), rs in sorted(groups.items()):
        def m(key):
            vals = [x[key] for x in rs if key in x]
            return np.mean(vals) if vals else float("nan")
        print(f"{b:<21}{v:<17}{len(rs):>3} {m('residual_l2'):>12.3e} {m('residual_share_within_0.1'):>16.3f} "
              f"{m('domain_area_share_within_0.1'):>13.4f} {m('residual_share_within_0.3'):>16.3f} "
              f"{m('mismatch_share_reentrant_edges'):>25.3f}")
    print("\n(residual shares and re-entrant-edge shares only exist for the L-shape; 'area share' is what a uniform"
          " distribution of the residual would give.)")


if __name__ == "__main__":
    main()

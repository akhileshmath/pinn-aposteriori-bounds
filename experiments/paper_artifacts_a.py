#!/usr/bin/env python3
"""Regenerate ALL tables, figures and quoted numbers of the a-energy study from the per-run JSON files.

    python experiments/paper_artifacts_a.py --runs results/a_energy_runs

Nothing here trains or evaluates a network: it only reads results/a_energy_runs/runs/*.json written by
experiments/run_a.py.  Every number that the manuscript quotes about these experiments should be copied
from paper/tables/a_*.tex or from results/paper/a_claims.json, never typed by hand.

Outputs
  paper/tables/a_theoremA_validation.tex, a_decomposition.tex, a_boundary_diagnostics.tex, a_loss_vs_estimator.tex
  paper/figures/a_fig_convergence.{pdf,png}, a_fig_decomposition.*, a_fig_lifting_vs_l2.*, a_fig_loss_vs_estimator.*
  results/paper/a_claims.json   (machine-readable facts + warnings, e.g. number of seeds behind every statement)
"""
import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.run_a import load_runs  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,           # embedded TrueType (publisher font checks)
})

# Validated categorical slots 1-3 (dataviz reference palette, light mode, all-pairs check passed).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#52514e"
BENCH_ORDER = ["poisson", "variable_coefficient", "l_shaped"]
BENCH_LABEL = {"poisson": "Poisson", "variable_coefficient": "Variable coefficient", "l_shaped": "L-shaped"}
VARIANT_ORDER = ["baseline", "wb1", "wb100", "sparse_bnd_16", "sparse_bnd_64", "tangent_g001", "tangent_g01",
                 "tangent_g1", "tangent_g10", "hardbc", "corner_refined", "corner_interior_only"]
VARIANT_TEX = {"baseline": r"baseline ($w_b{=}10$)", "wb1": r"$w_b{=}1$", "wb100": r"$w_b{=}100$",
               "sparse_bnd_16": r"16 fixed boundary pts", "sparse_bnd_64": r"64 fixed boundary pts",
               "tangent_g001": r"tangential, $\gamma{=}0.01$", "tangent_g01": r"tangential, $\gamma{=}0.1$",
               "tangent_g1": r"tangential, $\gamma{=}1$", "tangent_g10": r"tangential, $\gamma{=}10$",
               "hardbc": "hard BC", "corner_refined": "corner-refined (interior+boundary)",
               "corner_interior_only": "corner-refined (interior only)"}
VARIANT_SHORT = {"baseline": "baseline", "wb1": "w_b=1", "wb100": "w_b=100", "sparse_bnd_16": "16 bnd",
                 "sparse_bnd_64": "64 bnd", "tangent_g001": "γ=0.01", "tangent_g01": "γ=0.1", "tangent_g1": "γ=1",
                 "tangent_g10": "γ=10", "hardbc": "hard BC", "corner_refined": "corner", "corner_interior_only": "corner-int"}
GROUP = {"baseline": 0, "wb1": 0, "wb100": 0, "sparse_bnd_16": 0, "sparse_bnd_64": 0,
         "tangent_g001": 1, "tangent_g01": 1, "tangent_g1": 1, "tangent_g10": 1, "hardbc": 1,
         "corner_refined": 2, "corner_interior_only": 2}
GROUP_NAME = ["baseline, $w_b$ sweep, fixed boundary points", "tangential penalty / hard BC", "corner-refined collocation"]
GROUP_PLAIN = ["baseline, w_b sweep, fixed boundary pts", "tangential penalty / hard BC", "corner-refined collocation"]
GROUP_COLOR, GROUP_MARKER = [BLUE, ORANGE, AQUA], ["o", "s", "^"]


# ------------------------------------------------------------------------------------------- data
def flatten(r: Dict[str, Any]) -> Dict[str, Any]:
    ev, fs, last = r["evaluation"], r["fixed_sample"], r["evaluation"]["levels"][-1]
    l2 = ev["boundary"]["l2"]
    return {
        "benchmark": r["benchmark"], "variant": r["variant"], "seed": r["seed"], "quick": bool(r.get("quick")),
        "n_finest": last["n_cells"], "error_a": ev["true_error_a"], "error_grad": ev["true_error_grad"],
        "eta": last["eta"], "ratio": last["eta_over_true"], "dual": last["dual_norm"], "lifting": last["lifting_norm"],
        "ratio_extrap": r["extrapolation"].get("limit_over_true"), "order": r["extrapolation"].get("order"),
        "bnd_l2": l2, "bnd_tan": ev["boundary"]["tangential_h1"],
        "lift_per_l2": last["lifting_norm"] / l2 if l2 > 1e-12 else float("nan"),
        "res_loss": fs["residual_loss"], "bnd_loss": fs["boundary_loss"], "ref_selfconv": ev["ref_self_convergence_rel"],
        "levels": [(lv["n_cells"], lv["eta_over_true"]) for lv in ev["levels"]],
        "train_s": r["train_seconds"],
    }


def _ms(values) -> Tuple[float, float, int]:
    v = np.array([x for x in values if x is not None], dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else float("nan"), int(v.size)


def group_rows(rows):
    out: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        out.setdefault((r["benchmark"], r["variant"]), []).append(r)
    return out


def ordered_keys(groups):
    return [(b, v) for b in BENCH_ORDER for v in VARIANT_ORDER if (b, v) in groups]


# ------------------------------------------------------------------------------------------- tables
def _sci(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    if x == 0.0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10**e
    return f"${m:.2f}\\!\\times\\!10^{{{e}}}$"


def _pm_sci(mean: float, sd: float) -> str:
    if isinstance(mean, float) and math.isnan(mean):
        return "--"
    if mean == 0.0:
        return "0"
    if isinstance(sd, float) and math.isnan(sd):
        return _sci(mean)
    e = int(math.floor(math.log10(abs(mean))))
    return f"${mean / 10**e:.2f}\\pm{sd / 10**e:.2f}\\,\\times\\!10^{{{e}}}$"


def _pm_fix(mean: float, sd: float, nd: int = 4) -> str:
    if isinstance(mean, float) and math.isnan(mean):
        return "--"
    if isinstance(sd, float) and math.isnan(sd):
        return f"${mean:.{nd}f}$"
    return f"${mean:.{nd}f}\\pm{sd:.{nd}f}$"


def _write(lines: List[str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines + [""]))


def table_validation(groups, path):
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Benchmark & Variant & Seeds & Finest $n$ & $\eta_a/\|e\|_a$ & $\eta_a^{\mathrm{extr}}/\|e\|_a$ & $\max|\eta_a/\|e\|_a-1|$ \\",
             r"\midrule"]
    last_b = None
    for b, v in ordered_keys(groups):
        rs = groups[(b, v)]
        if last_b is not None and b != last_b:
            lines.append(r"\midrule")
        m, s, n = _ms([r["ratio"] for r in rs])
        me, se, _ = _ms([r["ratio_extrap"] for r in rs])
        worst = max(abs(r["ratio"] - 1.0) for r in rs)
        lines.append(f"{BENCH_LABEL[b] if b != last_b else ''} & {VARIANT_TEX[v]} & {n} & {rs[0]['n_finest']} & "
                     f"{_pm_fix(m, s)} & {_pm_fix(me, se)} & {worst:.1e} \\\\")
        last_b = b
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(lines, path)


def table_decomposition(groups, path):
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Benchmark & Variant & Seeds & $\|e\|_a$ & $\|R\|_{*,a}$ & $\|w_a\|_a$ & residual share of $\|e\|_a^2$ (\%) \\",
             r"\midrule"]
    last_b = None
    for b, v in ordered_keys(groups):
        rs = groups[(b, v)]
        if last_b is not None and b != last_b:
            lines.append(r"\midrule")
        share = [100.0 * r["dual"] ** 2 / (r["dual"] ** 2 + r["lifting"] ** 2) for r in rs]
        ms = _ms(share)
        lines.append(f"{BENCH_LABEL[b] if b != last_b else ''} & {VARIANT_TEX[v]} & {len(rs)} & "
                     f"{_pm_sci(*_ms([r['error_a'] for r in rs])[:2])} & {_pm_sci(*_ms([r['dual'] for r in rs])[:2])} & "
                     f"{_pm_sci(*_ms([r['lifting'] for r in rs])[:2])} & {_pm_fix(ms[0], ms[1], 2)} \\\\")
        last_b = b
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(lines, path)


def table_boundary(groups, path):
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Benchmark & Variant & Seeds & $\|g-\hat u\|_{L^2(\Gamma)}$ & $\|\partial_\tau(g-\hat u)\|_{L^2(\Gamma)}$ & $\|w_a\|_a/\|g-\hat u\|_{L^2(\Gamma)}$ \\",
             r"\midrule"]
    last_b = None
    for b, v in ordered_keys(groups):
        rs = groups[(b, v)]
        if last_b is not None and b != last_b:
            lines.append(r"\midrule")
        lines.append(f"{BENCH_LABEL[b] if b != last_b else ''} & {VARIANT_TEX[v]} & {len(rs)} & "
                     f"{_pm_sci(*_ms([r['bnd_l2'] for r in rs])[:2])} & {_pm_sci(*_ms([r['bnd_tan'] for r in rs])[:2])} & "
                     f"{_pm_fix(*_ms([r['lift_per_l2'] for r in rs])[:2], nd=2)} \\\\")
        last_b = b
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(lines, path)


def table_loss(groups, path):
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Benchmark & Variant & Seeds & $\|r\|_{L^2}$ (uniform sample) & $\|R\|_{*,a}$ & $\|R\|_{*,a}/\|r\|_{L^2}$ & $\|e\|_a$ \\",
             r"\midrule"]
    last_b = None
    for b, v in ordered_keys(groups):
        rs = groups[(b, v)]
        if last_b is not None and b != last_b:
            lines.append(r"\midrule")
        rl = [math.sqrt(r["res_loss"]) for r in rs]
        ratio = [r["dual"] / math.sqrt(r["res_loss"]) for r in rs]
        lines.append(f"{BENCH_LABEL[b] if b != last_b else ''} & {VARIANT_TEX[v]} & {len(rs)} & "
                     f"{_pm_sci(*_ms(rl)[:2])} & {_pm_sci(*_ms([r['dual'] for r in rs])[:2])} & "
                     f"{_pm_sci(*_ms(ratio)[:2])} & {_pm_sci(*_ms([r['error_a'] for r in rs])[:2])} \\\\")
        last_b = b
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(lines, path)


# ------------------------------------------------------------------------------------------- figures
def _clean_log(ax) -> None:
    """Readable log-axis ticks: 1-2-5 x 10^k over >= 1 decade; <= 4 plain decimals over a narrower range."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogFormatterSciNotation, LogLocator, NullFormatter, NullLocator

    for axis in (ax.xaxis, ax.yaxis):
        if axis.get_scale() != "log":
            continue
        lo, hi = sorted(axis.get_view_interval())
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())
        if hi / lo >= 10.0:
            axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=30))
            axis.set_major_formatter(LogFormatterSciNotation(labelOnlyBase=False))
            continue
        k0, k1 = int(math.floor(math.log10(lo))) - 1, int(math.ceil(math.log10(hi))) + 1
        cand = [m * 10.0**k for k in range(k0, k1 + 1) for m in range(1, 10) if lo < m * 10.0**k < hi]
        if len(cand) > 4:
            cand = [cand[i] for i in np.linspace(0, len(cand) - 1, 4).round().astype(int)]
        if cand:
            axis.set_major_locator(FixedLocator(cand))
            axis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:.2g}"))


def _save(fig, stem: str) -> None:
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png")
    plt.close(fig)


def _panels(rows, benches=BENCH_ORDER):
    return [b for b in benches if any(r["benchmark"] == b for r in rows)]


def fig_convergence(rows, stem):
    benches = _panels(rows)
    fig, axes = plt.subplots(2, len(benches), figsize=(3.6 * len(benches), 6.0), squeeze=False)
    for j, b in enumerate(benches):
        sel = [r for r in rows if r["benchmark"] == b]
        for r in sel:
            n = [x[0] for x in r["levels"]]
            y = np.array([x[1] for x in r["levels"]], dtype=float)
            axes[0, j].plot(n, y, color=BLUE, alpha=0.35, lw=1.0, marker="o", ms=3)
            axes[1, j].plot(n, np.abs(y - 1.0), color=BLUE, alpha=0.35, lw=1.0, marker="o", ms=3)
        axes[0, j].axhline(1.0, color=INK, lw=1.0, ls="--")
        axes[0, j].set_title(f"{BENCH_LABEL[b]} ({len(sel)} runs)", fontsize=10)
        axes[0, j].set_xscale("log", base=2)
        axes[1, j].set_xscale("log", base=2)
        axes[1, j].set_yscale("log")
        nref = np.array(sorted({x[0] for r in sel for x in r["levels"]}), dtype=float)
        ys = [abs(x[1] - 1.0) for r in sel for x in r["levels"] if abs(x[1] - 1.0) > 0]
        if len(nref) > 1 and ys:
            y0 = np.median([abs(r["levels"][0][1] - 1.0) for r in sel])
            p = 4.0 / 3.0 if b == "l_shaped" else 2.0
            axes[1, j].plot(nref, y0 * (nref / nref[0]) ** (-p), color=MUTED, lw=1.2, ls=":")
            axes[1, j].text(nref[-1], y0 * (nref[-1] / nref[0]) ** (-p), f"$h^{{{'4/3' if p < 2 else '2'}}}$", color=MUTED,
                            ha="right", va="bottom", fontsize=9)
        axes[1, j].set_xlabel("cells per side $n$")
    axes[0, 0].set_ylabel(r"$\eta_a/\|e\|_a$")
    axes[1, 0].set_ylabel(r"$|\eta_a/\|e\|_a-1|$")
    fig.suptitle("Theorem A: the computed estimator approaches the true a-energy error under mesh refinement", fontsize=10)
    fig.tight_layout()
    _save(fig, stem)


def fig_decomposition(groups, stem):
    benches = _panels([{"benchmark": b} for (b, _v) in groups])
    n_rows = max(sum(1 for v in VARIANT_ORDER if (b, v) in groups) for b in benches)
    fig, axes = plt.subplots(1, len(benches), figsize=(4.2 * len(benches), max(3.8, 0.33 * n_rows + 1.6)), squeeze=False)
    for j, b in enumerate(benches):
        ax = axes[0, j]
        vs = [v for v in VARIANT_ORDER if (b, v) in groups]
        for i, v in enumerate(vs):
            rs = groups[(b, v)]
            for key, col, off in (("dual", BLUE, -0.12), ("lifting", ORANGE, 0.12)):
                vals = np.array([r[key] for r in rs])
                if np.all(vals < 1e-12):      # exactly-zero lifting (hard BC): not drawn on a log axis
                    continue
                ax.plot(vals, np.full(vals.size, i + off), "o", ms=3, color=col, alpha=0.5, mec="none")
                ax.plot([vals.mean()], [i + off], "|", ms=14, mew=2.2, color=col)
            err = np.array([r["error_a"] for r in rs])
            ax.plot([err.mean()], [i], "D", ms=4.5, color=INK, mfc="white", mew=1.2)
        ax.set_xscale("log")
        _clean_log(ax)
        ax.set_yticks(range(len(vs)))
        ax.set_yticklabels([VARIANT_SHORT[v] for v in vs])
        ax.invert_yaxis()
        ax.set_title(BENCH_LABEL[b], fontsize=10)
        ax.set_xlabel("energy norm")
        if "hardbc" in vs:
            ax.annotate("lifting $\\equiv 0$", (0.98, vs.index("hardbc") + 0.12), xycoords=("axes fraction", "data"),
                        ha="right", fontsize=8, color=MUTED, va="center")
    handles = [plt.Line2D([], [], color=BLUE, marker="|", ls="", ms=10, mew=2, label=r"residual $\|R\|_{*,a}$"),
               plt.Line2D([], [], color=ORANGE, marker="|", ls="", ms=10, mew=2, label=r"boundary lifting $\|w_a\|_a$"),
               plt.Line2D([], [], color=INK, marker="D", mfc="white", ls="", ms=5, label=r"total $\|e\|_a$ (mean)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Where the energy error comes from (dots: single seeds, bars: seed mean)", fontsize=10)
    fig.tight_layout()
    _save(fig, stem)


def fig_lifting_vs_l2(rows, stem):
    benches = _panels(rows)
    fig, axes = plt.subplots(1, len(benches), figsize=(4.0 * len(benches), 3.8), squeeze=False)
    for j, b in enumerate(benches):
        ax = axes[0, j]
        sel = [r for r in rows if r["benchmark"] == b and r["bnd_l2"] > 1e-12]
        for g in range(3):
            pts = [r for r in sel if GROUP[r["variant"]] == g]
            if pts:
                ax.plot([r["bnd_l2"] for r in pts], [r["lifting"] for r in pts], GROUP_MARKER[g], color=GROUP_COLOR[g],
                        ms=5, alpha=0.8, mec="white", mew=0.5, ls="", label=GROUP_PLAIN[g])
        if sel:
            x = np.array([r["bnd_l2"] for r in sel])
            c = np.array([r["lift_per_l2"] for r in sel])
            xs = np.array([x.min() * 0.7, x.max() * 1.4])
            for q, ls in ((np.min(c), ":"), (np.max(c), "--")):
                ax.plot(xs, q * xs, color=MUTED, lw=1.0, ls=ls)
                ax.text(xs[1], q * xs[1], f" {q:.1f}", color=MUTED, fontsize=8, va="center")
        ax.set_xscale("log")
        ax.set_yscale("log")
        _clean_log(ax)
        ax.set_xlabel(r"$\|g-\hat u\|_{L^2(\Gamma)}$")
        ax.set_title(BENCH_LABEL[b], fontsize=10)
    axes[0, 0].set_ylabel(r"boundary lifting $\|w_a\|_a$")
    h, l = axes[0, -1].get_legend_handles_labels()
    if not h:
        h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Lifting energy vs. $L^2$ boundary mismatch (lines: smallest/largest ratio over runs)", fontsize=10)
    fig.tight_layout()
    _save(fig, stem)


def fig_loss_vs_estimator(groups, stem):
    benches = [b for b in BENCH_ORDER if any(k[0] == b for k in groups)]
    fig, axes = plt.subplots(2, len(benches), figsize=(3.9 * len(benches), 6.6), squeeze=False)
    for j, b in enumerate(benches):
        for i, (key, lab) in enumerate((("dual", r"residual norm $\|R\|_{*,a}$"), ("error_a", r"true error $\|e\|_a$"))):
            ax = axes[i, j]
            for v in [v for v in VARIANT_ORDER if (b, v) in groups and v != "hardbc"]:
                g = GROUP[v]
                rs = groups[(b, v)]
                x = np.array([math.sqrt(r["res_loss"]) for r in rs])
                y = np.array([r[key] for r in rs])
                ax.plot(x, y, GROUP_MARKER[g], color=GROUP_COLOR[g], ms=4, alpha=0.45, mec="none", ls="")
                ax.plot([x.mean()], [y.mean()], GROUP_MARKER[g], color=GROUP_COLOR[g], ms=8, mec=INK, mew=0.8, ls="")
                left = VARIANT_ORDER.index(v) % 2 == 1      # alternate sides to reduce label collisions
                ax.annotate(VARIANT_SHORT[v], (x.mean(), y.mean()), xytext=(-7 if left else 7, -10 if left else 5),
                            textcoords="offset points", fontsize=7.5, color=MUTED, ha="right" if left else "left")
            ax.set_xscale("log")
            ax.set_yscale("log")
            _clean_log(ax)
            ax.set_ylabel(lab)
            if i == 0:
                ax.set_title(BENCH_LABEL[b], fontsize=10)
            if i == 1:
                ax.set_xlabel(r"$\|r\|_{L^2}$ (fixed uniform sample)")
    handles = [plt.Line2D([], [], color=GROUP_COLOR[g], marker=GROUP_MARKER[g], ls="", ms=6, label=GROUP_PLAIN[g]) for g in range(3)]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Strong-form residual loss vs. energy-norm quantities (small: single seeds, large: seed mean; hard BC omitted)", fontsize=9.5)
    fig.tight_layout()
    _save(fig, stem)


# ------------------------------------------------------------------------------------------- claims
def build_claims(rows, groups) -> Dict[str, Any]:
    claims: Dict[str, Any] = {"n_runs": len(rows), "warnings": [], "per_benchmark": {}}
    seeds = sorted({r["seed"] for r in rows})
    claims["seeds"] = seeds
    if any(r["quick"] for r in rows):
        claims["warnings"].append("QUICK runs present: pipeline check only, do NOT quote these numbers.")
    if len(seeds) < 5:
        claims["warnings"].append(f"Only {len(seeds)} seed(s): report mean/sd and per-seed values, avoid significance language.")
    for b in BENCH_ORDER:
        if not any(r["benchmark"] == b for r in rows):
            continue
        rb = [r for r in rows if r["benchmark"] == b]
        d: Dict[str, Any] = {"n_runs": len(rb)}
        soft = [r for r in rb if r["variant"] != "hardbc"]
        d["theoremA_max_abs_ratio_deviation_finest"] = max(abs(r["ratio"] - 1.0) for r in rb)
        d["theoremA_max_abs_ratio_deviation_finest_softBC"] = max(abs(r["ratio"] - 1.0) for r in soft) if soft else None
        ext = [abs(r["ratio_extrap"] - 1.0) for r in rb if r["ratio_extrap"] is not None]
        d["extrapolated_max_abs_deviation"] = max(ext) if ext else None
        d["reference_selfconvergence_max_rel"] = max(r["ref_selfconv"] for r in rb)
        base = groups.get((b, "baseline"))
        if base:
            e0 = np.mean([r["error_a"] for r in base])
            d["baseline_residual_share_of_squared_error_pct"] = float(np.mean(
                [100 * r["dual"] ** 2 / (r["dual"] ** 2 + r["lifting"] ** 2) for r in base]))
            d["error_factor_baseline_over_variant"] = {
                v: float(e0 / np.mean([r["error_a"] for r in groups[(b, v)]]))
                for v in VARIANT_ORDER if (b, v) in groups and v != "baseline"}
            d["n_seeds_per_variant"] = {v: len(groups[(b, v)]) for v in VARIANT_ORDER if (b, v) in groups}
        c = [r["lift_per_l2"] for r in rb if not math.isnan(r["lift_per_l2"])]
        d["lifting_per_l2_min_max"] = [min(c), max(c)] if c else None
        d["dual_over_sqrt_fixed_residual_loss"] = {
            v: float(np.mean([r["dual"] / math.sqrt(r["res_loss"]) for r in groups[(b, v)]]))
            for v in VARIANT_ORDER if (b, v) in groups}
        claims["per_benchmark"][b] = d
    claims["notes"] = [
        "Factors are ratios of seed means; with few seeds they are descriptive, not tested.",
        "All analyses in a_claims.json were chosen after seeing the data (exploratory) unless pre-registered in the manuscript plan.",
    ]
    return claims


# ------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.path.join(PROJECT_ROOT, "results", "a_energy_runs"))
    ap.add_argument("--tables-dir", default=os.path.join(PROJECT_ROOT, "paper", "tables"))
    ap.add_argument("--figure-dir", default=os.path.join(PROJECT_ROOT, "paper", "figures"))
    ap.add_argument("--claims-file", default=os.path.join(PROJECT_ROOT, "results", "paper", "a_claims.json"))
    args = ap.parse_args(argv)

    rows = [flatten(r) for r in load_runs(args.runs)]
    if not rows:
        raise SystemExit(f"No runs in {args.runs}/runs - run experiments/run_a.py first.")
    groups = group_rows(rows)
    t, f = args.tables_dir, args.figure_dir
    table_validation(groups, os.path.join(t, "a_theoremA_validation.tex"))
    table_decomposition(groups, os.path.join(t, "a_decomposition.tex"))
    table_boundary(groups, os.path.join(t, "a_boundary_diagnostics.tex"))
    table_loss(groups, os.path.join(t, "a_loss_vs_estimator.tex"))
    fig_convergence(rows, os.path.join(f, "a_fig_convergence"))
    fig_decomposition(groups, os.path.join(f, "a_fig_decomposition"))
    fig_lifting_vs_l2(rows, os.path.join(f, "a_fig_lifting_vs_l2"))
    fig_loss_vs_estimator(groups, os.path.join(f, "a_fig_loss_vs_estimator"))
    claims = build_claims(rows, groups)
    os.makedirs(os.path.dirname(args.claims_file), exist_ok=True)
    with open(args.claims_file, "w", encoding="utf-8") as handle:
        json.dump(claims, handle, indent=2, default=float)
    print(f"{len(rows)} runs -> 4 tables in {t}, 4 figures in {f}, claims in {args.claims_file}")
    for w in claims["warnings"]:
        print("WARNING:", w)


if __name__ == "__main__":
    main()

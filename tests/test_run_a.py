"""Tests of the multi-seed driver (quick mode) and of the convergence-order helper."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.a_energy import observed_order_and_limit  # noqa: E402


def test_observed_order_recovers_synthetic_limit():
    p, c, lim = 4.0 / 3.0, 0.7, 1.0
    etas = [lim + c * (1.0 / n) ** p for n in (64, 128, 256, 512)]
    out = observed_order_and_limit(etas)
    assert abs(out["order"] - p) < 1e-9 and abs(out["limit"] - lim) < 1e-12
    out = observed_order_and_limit(etas, assumed_order=p)
    assert abs(out["limit_assumed"] - lim) < 1e-12


def test_observed_order_rejects_non_monotone_or_short():
    assert observed_order_and_limit([1.0, 2.0])["limit"] is None
    assert observed_order_and_limit([1.0, 1.5, 1.2])["limit"] is None       # not monotone
    assert observed_order_and_limit([1.0, 1.2, 1.5])["limit"] is None       # diverging


torch = pytest.importorskip("torch")
from experiments import run_a  # noqa: E402


def _task(tmp_path, b, v, seed=0, **kw):
    t = {"benchmark": b, "variant": v, "seed": seed, "out": str(tmp_path), "dtype": "float64",
         "ladder": None, "quick": True, "force": False, "threads": 1}
    t.update(kw)
    return t


def test_enumerate_runs_respects_applicability():
    runs = run_a.enumerate_runs(["poisson", "l_shaped"], ["baseline", "hardbc", "corner_refined"], [0])
    assert ("l_shaped", "hardbc", 0) not in runs and ("poisson", "corner_refined", 0) not in runs
    assert ("poisson", "hardbc", 0) in runs and ("l_shaped", "corner_refined", 0) in runs
    with pytest.raises(ValueError):
        run_a.build_run_config("l_shaped", "hardbc", True)


def test_variant_configs_are_merged_correctly():
    base = run_a.build_run_config("poisson", "baseline", False)["training"]
    assert base["w_b"] == 10.0 and base.get("w_tangent", 0.0) == 0.0
    assert run_a.build_run_config("poisson", "wb100", False)["training"]["w_b"] == 100.0
    assert run_a.build_run_config("l_shaped", "tangent_g10", False)["training"]["w_tangent"] == 10.0
    assert run_a.build_run_config("poisson", "hardbc", False)["network"]["hard_bc"] == "unit_square_zero"
    s = run_a.build_run_config("l_shaped", "corner_refined", False)["sampler"]
    assert s["interior"]["corner_fraction"] == 0.5 and s["boundary"]["corner_fraction"] == 0.3
    # the original benchmark config is untouched
    assert run_a.build_run_config("l_shaped", "baseline", False)["training"]["adam_epochs"] == 8000


def test_quick_runs_write_complete_records_and_resume(tmp_path):
    for b, v in [("poisson", "baseline"), ("poisson", "hardbc"), ("l_shaped", "tangent_g1"), ("l_shaped", "corner_refined")]:
        res = run_a.run_one(_task(tmp_path, b, v))
        assert res["status"] == "done", (b, v, open(tmp_path / "errors" / f"{b}__{v}__seed0.txt").read()
                                         if (tmp_path / "errors").exists() else res)
    rec = json.load(open(tmp_path / "runs" / "poisson__hardbc__seed0.json"))
    for key in ("config", "history", "fixed_sample", "evaluation", "extrapolation", "git_commit", "train_seconds"):
        assert key in rec
    ev = rec["evaluation"]
    assert ev["levels"][-1]["lifting_norm"] < 1e-12            # hard BC => zero lifting
    assert ev["boundary"]["l2"] < 1e-12
    assert os.path.exists(tmp_path / "weights" / "poisson__hardbc__seed0.pt")
    rec_l = json.load(open(tmp_path / "runs" / "l_shaped__tangent_g1__seed0.json"))
    assert rec_l["config"]["training"]["w_tangent"] == 1.0 and max(rec_l["history"]["tangential_loss"]) > 0
    assert rec_l["evaluation"]["ref_method"] == "graded_polar"
    # resume: second call skips, force recomputes
    assert run_a.run_one(_task(tmp_path, "poisson", "baseline"))["status"] == "skipped"
    assert run_a.run_one(_task(tmp_path, "poisson", "baseline", force=True))["status"] == "done"


def test_same_seed_is_reproducible(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    run_a.run_one(_task(a, "poisson", "baseline", seed=5))
    run_a.run_one(_task(b, "poisson", "baseline", seed=5))
    ra = json.load(open(a / "runs" / "poisson__baseline__seed5.json"))
    rb = json.load(open(b / "runs" / "poisson__baseline__seed5.json"))
    assert ra["history"]["total_loss"] == rb["history"]["total_loss"]
    assert ra["evaluation"]["true_error_a"] == rb["evaluation"]["true_error_a"]


def test_failure_is_recorded_not_raised(tmp_path):
    res = run_a.run_one(_task(tmp_path, "l_shaped", "hardbc"))      # not applicable -> ValueError inside
    assert res["status"] == "failed" and (tmp_path / "errors" / "l_shaped__hardbc__seed0.txt").exists()


def test_summarize_prints_and_writes_csv(tmp_path, capsys):
    for seed in (0, 1):
        run_a.run_one(_task(tmp_path, "poisson", "baseline", seed=seed))
    run_a.summarize(str(tmp_path))
    out = capsys.readouterr().out
    assert "poisson" in out and "baseline" in out and "quick" in out.lower()
    assert (tmp_path / "summary.csv").exists()

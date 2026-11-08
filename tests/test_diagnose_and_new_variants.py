"""Tests for the root-cause diagnostics (experiments/diagnose_runs.py) and the opt-in variants."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")
from experiments import diagnose_runs, run_a  # noqa: E402

NEW = ["tangent_g01", "tangent_g001", "corner_interior_only", "sparse_bnd_16", "sparse_bnd_64"]


def test_new_variants_configs_full_mode():
    assert run_a.build_run_config("poisson", "tangent_g01", False)["training"]["w_tangent"] == 0.1
    assert run_a.build_run_config("l_shaped", "tangent_g001", False)["training"]["w_tangent"] == 0.01
    s = run_a.build_run_config("l_shaped", "corner_interior_only", False)["sampler"]
    assert s["interior"]["corner_fraction"] == 0.5 and "corner_fraction" not in s.get("boundary", {})
    t = run_a.build_run_config("poisson", "sparse_bnd_16", False)["training"]
    assert t["n_boundary"] == 16 and t["resample_every"] == 0
    with pytest.raises(ValueError):
        run_a.build_run_config("poisson", "corner_interior_only", False)


def test_variant_overrides_survive_quick_mode():
    """Regression: quick defaults must not clobber a variant's own overrides."""
    t = run_a.build_run_config("l_shaped", "sparse_bnd_64", True)["training"]
    assert t["n_boundary"] == 64 and t["resample_every"] == 0
    assert t["adam_epochs"] == 150                       # quick defaults still apply elsewhere
    assert run_a.build_run_config("poisson", "baseline", True)["training"]["n_boundary"] == 100


def test_original_variants_unchanged_by_new_ones():
    # adding variants must not alter the original 7 configs
    assert run_a.build_run_config("poisson", "baseline", False)["training"]["w_b"] == 10.0
    assert run_a.build_run_config("poisson", "tangent_g1", False)["training"]["w_tangent"] == 1.0


@pytest.fixture(scope="module")
def runs_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("diag")
    todo = [("poisson", "baseline", 0), ("l_shaped", "baseline", 0), ("l_shaped", "sparse_bnd_16", 0)]
    for b, v, s in todo:
        run_a.run_one({"benchmark": b, "variant": v, "seed": s, "out": str(out), "dtype": "float64",
                       "ladder": [8, 16, 32], "quick": True, "force": False, "threads": 1})
    return str(out)


def test_diagnostics_consistency(runs_dir):
    diagnose_runs.main(["--runs", runs_dir])
    diag = json.load(open(os.path.join(runs_dir, "diagnostics.json")))
    assert len(diag) == 3
    recs = {r["id"]: r for r in run_a.load_runs(runs_dir)}
    for d in diag:
        # edge sums reproduce the independently computed boundary L2 norm of the run record
        ref = recs[d["id"]]["evaluation"]["boundary"]["l2"]
        assert abs(d["boundary_l2"] - ref) <= 1e-5 * ref
        assert abs(np.sqrt(sum(d["boundary_l2_sq_by_edge"].values())) - d["boundary_l2"]) < 1e-14
        assert d["residual_l2"] > 0 and np.isfinite(d["residual_l2"])
        if d["benchmark"] == "l_shaped":
            for k in ("residual_share_within_0.1", "residual_share_within_0.3", "mismatch_share_reentrant_edges"):
                assert 0.0 <= d[k] <= 1.0
            assert d["residual_share_within_0.1"] <= d["residual_share_within_0.3"] + 1e-12
            assert len(d["boundary_l2_sq_by_edge"]) == 6
        else:
            assert len(d["boundary_l2_sq_by_edge"]) == 4

#!/usr/bin/env python3
"""Sanity checks for experiment configuration loading and result schemas."""

import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.common import load_benchmark_configs, validate_benchmark_results


def main() -> int:
    config_path = os.path.join(PROJECT_ROOT, "configs", "validated_benchmarks.json")
    configs = load_benchmark_configs(config_path)
    assert set(configs.keys()) == {"poisson", "variable_coefficient", "l_shaped"}

    sample_results = [
        {
            "benchmark": "Synthetic",
            "key": "synthetic",
            "description": "Schema validation sample",
            "architecture": {"hidden_dims": [8, 8], "activation": "tanh"},
            "training_config": {"adam_epochs": 1},
            "history": {
                "epochs": [1],
                "total_loss": [1.0],
                "residual_loss": [0.5],
                "boundary_loss": [0.5],
            },
            "metrics": {
                "estimated_error_energy": 1.0,
                "true_error_energy": 0.9,
                "residual_contribution": 0.1,
                "boundary_lifting_norm": 0.9,
                "effectivity": 1.111111,
                "training_loss": 1.0,
                "mesh_history": [],
            },
        }
    ]
    validate_benchmark_results(sample_results)
    print("Schema checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

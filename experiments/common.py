#!/usr/bin/env python3
"""Shared utilities for validated experiments and paper artifacts."""

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

import numpy as np
import torch

from src.pinn import TrainingConfig


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)


def _git_commit(project_root: str) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def collect_run_metadata(
    *,
    project_root: str,
    entrypoint: str,
    seed: int,
    device: str,
    config_path: str | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    metadata = {
        "entrypoint": entrypoint,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "seed": seed,
        "device": device,
    }
    if config_path is not None:
        metadata["config_path"] = os.path.abspath(config_path)
    if extra:
        metadata.update(extra)
    return metadata


def load_benchmark_configs(config_path: str) -> Dict[str, Dict[str, Any]]:
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    configs: Dict[str, Dict[str, Any]] = {}
    for key, config in raw.items():
        configs[key] = {
            "network": config["network"],
            "training": TrainingConfig(**config["training"]),
            "mesh_size": int(config["mesh_size"]),
            "sampler": config.get("sampler", {}),
        }
    return configs


def _validate_keys(name: str, payload: Dict[str, Any], required: Iterable[str]) -> None:
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{name} is missing required keys: {missing}")


def validate_benchmark_result(result: Dict[str, Any]) -> None:
    _validate_keys(
        "benchmark result",
        result,
        ["benchmark", "key", "description", "architecture", "training_config", "history", "metrics"],
    )
    _validate_keys(
        "history",
        result["history"],
        ["epochs", "total_loss", "residual_loss", "boundary_loss"],
    )
    _validate_keys(
        "metrics",
        result["metrics"],
        [
            "estimated_error_energy",
            "true_error_energy",
            "residual_contribution",
            "boundary_lifting_norm",
            "effectivity",
            "training_loss",
            "mesh_history",
        ],
    )


def validate_benchmark_results(results: List[Dict[str, Any]]) -> None:
    if not isinstance(results, list) or not results:
        raise ValueError("validated results must be a non-empty list")
    for result in results:
        validate_benchmark_result(result)


def validate_paper_artifact_payload(payload: Dict[str, Any]) -> None:
    _validate_keys("paper artifact payload", payload, ["metadata", "source_results", "benchmarks", "ablation", "convergence"])
    validate_benchmark_results(payload["benchmarks"])


def serialise_training_config(config: Any) -> Dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    return dict(config)

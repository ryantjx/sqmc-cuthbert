"""Validate the output artifacts of the rbsqmc model_unbiased Colab pipeline.

Checks that every required artifact exists and is non-empty, that the JSON
artifacts parse and contain only finite numbers, and that the optimization
summary satisfies the acceptance criteria (improvement over baseline, histories
matching ``n_epochs``).

Usage:
    python rbsqmc/comparison/sqmc_smc/validate_model_unbiased_outputs.py <output_dir>

Uses only the standard library (plus numpy if available for .npz / array
checks). Exits non-zero if any check fails.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

# Artifacts that must be present (paths relative to the output dir).
REQUIRED_FILES = [
    "params_unbiased.json",
    "optimization_summary.json",
    "optimization_logZ_curve.png",
    "gradient_norm_curve.png",
    "run_config.json",
    "filtered/filter_states.npz",
    "filtered/timeseries_states.json",
    "filtered/final_rankings.png",
    "filtered/timeseries_states.png",
    "filtered/top_strengths.png",
    "predict/predictions.json",
    "predict/post_prediction_filter_rankings.json",
]

# JSON files that must parse and contain only finite numbers.
JSON_FILES = [
    "params_unbiased.json",
    "optimization_summary.json",
    "run_config.json",
]


def _finite_values(node) -> list[str]:
    """Recursively collect (path, value) for every non-finite number."""
    bad = []

    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{prefix}.{k}" if prefix else str(k))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{prefix}[{i}]")
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            import math
            if not math.isfinite(float(obj)):
                bad.append(f"{prefix}={obj!r}")

    walk(node)
    return bad


def _load_json(path):
    import json
    with open(path, "r") as f:
        return json.load(f)


def validate(output_dir: str) -> None:
    root = Path(output_dir)

    # 1. All required artifacts present and non-empty.
    for rel in REQUIRED_FILES:
        p = root / rel
        if not p.is_file() or p.stat().st_size == 0:
            raise ValueError(f"required artifact missing or empty: {p}")

    # 2. JSON files parse and contain only finite numbers.
    for rel in JSON_FILES:
        data = _load_json(root / rel)
        bad = _finite_values(data)
        if bad:
            raise ValueError(f"{rel} contains non-finite values: {bad[:5]}")

    # 3. params have finite kappa / alpha / beta.
    params = _load_json(root / "params_unbiased.json")
    for key in ("kappa", "alpha", "beta"):
        if key not in params:
            raise ValueError(f"params_unbiased.json missing key: {key}")

    # 4. Optimization summary acceptance criteria.
    summary = _load_json(root / "optimization_summary.json")
    for key in ("baseline_logZ", "best_logZ", "final_filter_logZ"):
        if key not in summary:
            raise ValueError(f"optimization_summary.json missing key: {key}")

    n_epochs = int(summary.get("n_epochs", 0))
    if n_epochs <= 0:
        raise ValueError("optimization_summary.json n_epochs must be positive")

    for key in ("train_logZ_history", "test_logZ_history", "gradient_norm_history"):
        hist = summary.get(key)
        if not isinstance(hist, list):
            raise ValueError(f"optimization_summary.json missing list: {key}")
        if len(hist) != n_epochs:
            raise ValueError(
                f"{key} length {len(hist)} != n_epochs {n_epochs}"
            )

    if not (summary["best_logZ"] > summary["baseline_logZ"]):
        raise ValueError(
            f"optimization did not improve: best_logZ {summary['best_logZ']} "
            f"<= baseline_logZ {summary['baseline_logZ']}"
        )

    # 5. Run config parses and carries the required knobs.
    run_cfg = _load_json(root / "run_config.json")
    for key in ("training_start_date", "test_start_date", "prediction_start_date",
                "n_particles", "max_goals", "n_epochs", "learning_rate", "n_reps"):
        if key not in run_cfg:
            raise ValueError(f"run_config.json missing key: {key}")

    print(f"Validation passed: {len(REQUIRED_FILES)} artifacts OK in {output_dir}")


def main(argv=None) -> int:
    if len(sys.argv[1:]) != 1:
        print(__doc__)
        return 2
    try:
        validate(sys.argv[1])
    except (ValueError, OSError, KeyError) as error:
        print(f"Validation FAILED: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

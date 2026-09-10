"""Validate downloaded numerical arrays, timing summaries and PNG figures."""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from comparison_protocol import read_json


def validate_numerical(root, stage, config):
    root = Path(root)
    args = config[stage]
    rows = read_json(root / "results.json")
    for path in root.glob("*.png"):
        with Image.open(path) as figure:
            figure.verify()
        with Image.open(path) as figure:
            if min(figure.size) < 100:
                raise ValueError("Unexpectedly small figure")
    for row in rows:
        if not np.allclose([row["q25_seconds"], row["q75_seconds"]], np.quantile(row["samples_seconds"], [.25, .75]), rtol=1e-12, atol=0):
            raise ValueError("Incorrect timing quartiles")
    comparisons = read_json(root / "cpu_gpu_comparison.json")
    if stage in {"qmc", "hilbert_sort"}:
        fields = ["sequence", "dimension", "n"] + (["mode"] if stage == "qmc" else [])
        expected = {tuple(row[k] for k in fields) for row in rows}
        if len(comparisons) != len(expected) or {tuple(c[k] for k in fields) for c in comparisons} != expected:
            raise ValueError("Incomplete CPU/GPU comparison grid")
        for comparison in comparisons:
            pair = {r["backend"]: r for r in rows if r.get("implementation", "jax") == "jax" and all(r[k] == comparison[k] for k in fields)}
            if not np.isclose(comparison["cpu_over_gpu"], pair["cpu"]["median_seconds"] / pair["gpu"]["median_seconds"]):
                raise ValueError("Incorrect speedup ratio")
    if stage == "qmc" and "implementations" in args:
        scipy_pairs = read_json(root / "scipy_comparison.json")
        expected_scipy = {key + (backend,) for key in expected for backend in args["platforms"]}
        if len(scipy_pairs) != len(expected_scipy) or {tuple(c[k] for k in fields) + (c["jax_backend"],) for c in scipy_pairs} != expected_scipy:
            raise ValueError("Incomplete SciPy comparison grid")
        for c in scipy_pairs:
            matched = {(r["implementation"], r["backend"]): r for r in rows if all(r[k] == c[k] for k in fields)}
            scipy = matched[("scipy", "cpu")]
            jax = matched[("jax", c["jax_backend"])]
            if not np.isclose(c["scipy_over_jax"], scipy["median_seconds"] / jax["median_seconds"], rtol=1e-12, atol=0):
                raise ValueError("Incorrect SciPy speedup ratio")
            if scipy["repetition_seeds"] != jax["repetition_seeds"] or len(set(scipy["repetition_seeds"])) != args["repeats"]:
                raise ValueError("Invalid QMC repetition seeds")
    if stage == "hilbert_sort":
        for sequence in args["sequences"]:
            for dimension in args["dimensions"]:
                for n in args["n_values"]:
                    with np.load(root / f"{sequence}_d{dimension}_n{n}_inputs.npz", allow_pickle=False) as saved:
                        points = saved["points"]
                        if points.shape != (n, dimension) or not np.isfinite(points).all():
                            raise ValueError("Invalid Hilbert input array")
                        digest = hashlib.sha256(points.tobytes()).hexdigest()
                        if any(r["input_sha256"] != digest for r in rows if r["sequence"] == sequence and r["dimension"] == dimension and r["n"] == n):
                            raise ValueError("CPU/GPU Hilbert input hashes differ from saved array")
    if stage != "sqmc":
        return
    records = read_json(root / "accuracy_records.json")
    expected_count = len(rows) * args["datasets"] * (args["selection_reps"] + args["validation_reps"])
    if len(records) != expected_count:
        raise ValueError("Incomplete accuracy records")
    for row in rows:
        dimension, n, backend = row["dimension"], row["n"], row["backend"]
        with np.load(root / f"reference_d{dimension}.npz", allow_pickle=False) as reference:
            truth, variance = reference["means"], reference["variances"]
            shape = (args["datasets"], args["n_steps"], dimension)
            if truth.shape != shape or variance.shape != shape or not np.isfinite(truth).all() or not (variance > 0).all():
                raise ValueError("Invalid Kalman reference")
            for phase in ("selection", "validation"):
                repetitions = args[phase + "_reps"]
                selected = [r for r in records if r["dimension"] == dimension and r["n"] == n and r["backend"] == backend and r["phase"] == phase]
                if {(r["dataset"], r["replicate"]) for r in selected} != {(d, r) for d in range(args["datasets"]) for r in range(repetitions)} or len(selected) != args["datasets"] * repetitions:
                    raise ValueError("Incomplete or duplicate accuracy replicates")
                with np.load(root / f"{backend}_d{dimension}_n{n}_{phase}.npz", allow_pickle=False) as saved:
                    estimates = saved["means"]
                    if estimates.shape != (args["datasets"] * repetitions, args["n_steps"], dimension) or not np.isfinite(estimates).all():
                        raise ValueError("Invalid SQMC estimates")
                    for record in selected:
                        dataset, replicate = record["dataset"], record["replicate"]
                        error = np.mean((estimates[dataset * repetitions + replicate] - truth[dataset])**2 / variance[dataset])
                        if not np.isclose(error, record["normalized_mse"], rtol=1e-12, atol=1e-14):
                            raise ValueError("Saved SQMC estimates disagree with reported error")
                reported = row[phase + "_accuracy"]
                if not np.isclose(reported["rmse"], np.sqrt(np.mean([r["normalized_mse"] for r in selected])), rtol=1e-12):
                    raise ValueError("Incorrect aggregate accuracy")
                interval = reported["rmse_ci95"]
                if len(interval) != 2 or not np.isfinite(interval).all() or not 0 <= interval[0] <= interval[1]:
                    raise ValueError("Invalid bootstrap interval")
    for summary in read_json(root / "budget_summary.json"):
        if summary["status"] == "selected":
            chosen = next(r for r in rows if all(r[k] == summary[k] for k in ("backend", "dimension", "n")))
            if summary["measured_fraction_within_budget"] != np.mean(np.asarray(chosen["samples_seconds"]) <= summary["budget_seconds"]):
                raise ValueError("Incorrect measured budget compliance")

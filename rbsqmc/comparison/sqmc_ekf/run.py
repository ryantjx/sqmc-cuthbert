"""CLI entrypoint that wires the complete SQMC–EKF comparison pipeline.

Runs data loading, training, prediction, evaluation, plots, and the report
into a single timestamped run directory under ``outputs/DDMMYYYY_HHMM/``.

Usage:
    python -m rbsqmc.comparison.sqmc_ekf.run \
        --config rbsqmc/comparison/sqmc_ekf/scripts/config/config_gpu.json \
        --data rbsqmc/data/results.csv \
        [--smoke] [--output-dir ...]
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import jax
import numpy as np

from rbsqmc.src.data.data_ekf import load_dataset as data_mod_load_dataset
from rbsqmc.comparison.sqmc_ekf.scripts import evaluate as eval_mod
from rbsqmc.comparison.sqmc_ekf.scripts import plots as plots_mod
from rbsqmc.comparison.sqmc_ekf.scripts import predict as predict_mod
from rbsqmc.comparison.sqmc_ekf.scripts import report as report_mod
from rbsqmc.comparison.sqmc_ekf.scripts import train
from rbsqmc.comparison.sqmc_ekf.scripts.scalar_table import (
    latex_escape as _latex_escape, table_body as _scalar_table_body,
)
from rbsqmc.comparison.sqmc_ekf.scripts.scaling import sqmc_match_scales
from rbsqmc.src.model.rbsqmc.model_rbsqmc import run_filter_sqmc
from rbsqmc.comparison.sqmc_ekf.scripts import diagnostics as diag_mod


DEFAULT_DATA = "rbsqmc/data/results.csv"


def _output_dir(base):
    stamp = datetime.now().strftime("%d%m%Y_%H%M")
    out = os.path.join(base, stamp)
    os.makedirs(out, exist_ok=True)
    return out


def _save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(value, f, indent=2, default=lambda o: o.tolist())


def _metrics_flat(metrics):
    """Serialize a Metrics object (all + worldcup) to a dict."""
    return {"all": metrics.all, "worldcup": metrics.worldcup}


def _config_digest(config):
    """SHA-256 of the canonicalized config, matching sqmc/comparison's protocol."""
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _git_commit():
    """Return the current HEAD revision, or None if not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _file_sha256(path):
    """SHA-256 of a file's bytes, or None if the file is missing."""
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def compute_setup(config):
    """Collect host/device/software setup info, mirroring the reference run_config."""
    devices = {}
    for backend in ("cpu", "gpu"):
        try:
            devices[backend] = [
                {"kind": d.device_kind, "id": d.id, "platform": d.platform}
                for d in jax.devices(backend)
            ]
        except Exception:
            devices[backend] = []
    try:
        import jaxlib
        jaxlib_version = jaxlib.__version__
    except Exception:
        jaxlib_version = None

    gpu = None
    nvidia_smi = None
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        nvidia_smi = subprocess.check_output(["nvidia-smi"], text=True)
    except Exception:
        pass

    cpu_models = []
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        cpu_models = sorted({
            line.split(":", 1)[1].strip()
            for line in cpuinfo.splitlines()
            if line.startswith("model name")
        })
    except Exception:
        cpu_models = [platform.processor()] if platform.processor() else []

    host_memory = None
    try:
        host_memory = Path("/proc/meminfo").read_text()
    except Exception:
        pass

    return {
        "source_commit": _git_commit(),
        "run_id": datetime.now().strftime("%d%m%Y_%H%M"),
        "config_sha256": _config_digest(config),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cpu_models": cpu_models,
        "host_memory": host_memory,
        "gpu_name_memory_MiB_driver": gpu,
        "nvidia_smi": nvidia_smi,
        "devices": devices,
        "execution_backend": jax.default_backend(),
        "jax": jax.__version__,
        "jaxlib": jaxlib_version,
        "versions": {
            d.metadata["Name"]: d.version
            for d in importlib.metadata.distributions()
            if d.metadata["Name"]
        },
    }


def run(cfg, data_path, smoke=False, output_dir=None, methods=("ekf", "sqmc"),
        evaluate_params=None):
    """Execute the comparison for the requested methods and return a results dict.

    When ``output_dir`` is given it is used as the exact run directory (no
    timestamp subdir is appended), so the Colab worker can point it at the VM
    run root and the ``results/``/``images/`` subfolders land directly there.

    ``methods`` selects which methods to train/predict/evaluate. A single-method
    run persists that method's artifacts (predictions, metrics, history, images)
    but skips the combined report/plots/CSVs; use :func:`combine` to merge an
    EKF run and an SQMC run into a complete comparison. When ``evaluate_params``
    is supplied, training is skipped and the saved checkpoint is replayed.
    """
    # A stored ``smoke`` key in the config is authoritative so that a launcher
    # writing the effective config to disk reproduces the same subset as the
    # explicit --smoke CLI flag.
    smoke = smoke or bool(cfg.get("smoke"))
    requested = os.environ.get("RBSQMC_PLATFORM")
    if requested in {"cpu", "cuda"}:
        expected = "gpu" if requested == "cuda" else "cpu"
        if jax.default_backend() != expected:
            raise RuntimeError(f"Requested {expected} execution, got {jax.default_backend()}")
    dataset = data_mod_load_dataset(data_path, cfg, smoke=smoke)
    if output_dir:
        run_dir = output_dir
        os.makedirs(run_dir, exist_ok=True)
    else:
        base_out = os.path.join(os.path.dirname(__file__), "outputs")
        run_dir = _output_dir(base_out)
    # Separate machine-readable results (JSON/CSV/MD) from images (PNG).
    results_dir = os.path.join(run_dir, "results")
    images_dir = os.path.join(run_dir, "images")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Persist the exact config + dataset metadata for reproducibility.
    _save_json(os.path.join(results_dir, "comparison_config.json"), cfg)
    _save_json(os.path.join(results_dir, "run_config.json"), compute_setup(cfg))
    _save_json(os.path.join(results_dir, "dataset_metadata.json"), dataset.metadata)

    if evaluate_params is not None and tuple(methods) != ("sqmc",):
        raise ValueError("--evaluate-params currently requires --methods sqmc")
    methods_obj = None if evaluate_params is not None else train.Methods(dataset, cfg)
    root = jax.random.PRNGKey(cfg["seed"])

    results = {"cfg": cfg}
    for name in methods:
        saved_checkpoint = None
        if evaluate_params is not None:
            saved_checkpoint = _resolve_checkpoint(evaluate_params, name)
            destination = Path(results_dir) / name / "fitted_params.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(saved_checkpoint, destination)
            _save_json(Path(results_dir) / "evaluation_metadata.json", {
                "evaluation_only": True,
                "parameter_source": str(saved_checkpoint),
                "parameter_checkpoint": _read_json(saved_checkpoint).get("checkpoint_epoch"),
            })
        results[name] = _run_method(
            name, methods_obj, dataset, cfg, root, results_dir, images_dir,
            saved_checkpoint=saved_checkpoint,
        )
        # Persist per-method artifacts so a partial run can be combined later.
        _save_json(os.path.join(results_dir, f"{name}_predictions.json"), results[name]["records"])
        _save_json(os.path.join(results_dir, f"{name}_metrics.json"), _metrics_flat(results[name]["metrics"]))
        _save_json(os.path.join(results_dir, f"{name}_history.json"), results[name]["history"])
        results[name]["summary"]["prediction_sec"] = results[name]["prediction_sec"]
        _save_json(os.path.join(results_dir, name, "summary.json"), results[name]["summary"])
        plots_mod.plot_prediction(results[name]["records"], cfg["max_goals"],
                                  os.path.join(images_dir, f"{name}_predictions"))

    if set(methods) == {"ekf", "sqmc"}:
        _write_combined(results_dir, images_dir, dataset, results, cfg, run_dir)
    else:
        from rbsqmc.comparison.sqmc_ekf.scripts.sqmc_ekf_protocol import validate_run
        validate_run(run_dir, cfg, methods=methods)
        print(f"Partial comparison ({', '.join(methods)}) complete: {run_dir}", flush=True)
    return results, run_dir


def _resolve_checkpoint(path, method):
    """Resolve a saved method checkpoint for evaluation-only replay."""
    path = Path(path)
    candidates = [
        path if path.is_file() else None,
        path / "fitted_params.json" if path.is_dir() else None,
        path / method / "fitted_params.json" if path.is_dir() else None,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            payload = _read_json(candidate)
            if payload.get("method") != method:
                raise ValueError(f"Checkpoint {candidate} is not for method {method!r}")
            return candidate
    raise FileNotFoundError(
        f"Could not find {method} fitted_params.json under {path}"
    )


def _write_combined(results_dir, images_dir, dataset, results, cfg, run_dir):
    """Write the combined report, overlay plot, CSVs, metadata and validate."""
    plots_mod.plot_convergence(
        results["ekf"]["history"], results["sqmc"]["history"], images_dir
    )
    report_mod.write_report(results_dir, dataset, results)
    _save_json(os.path.join(results_dir, "summary.json"), {
        "ekf": results["ekf"]["summary"], "sqmc": results["sqmc"]["summary"]})
    write_performance_metrics_csv(results_dir, results)
    write_logz_history_csv(results_dir, results)
    _write_scalar_comparison(results_dir, cfg, run_dir)
    _save_json(os.path.join(results_dir, "run_metadata.json"), {
        "completed_at_utc": datetime.utcnow().isoformat() + "Z",
        "run_id": os.path.basename(run_dir),
        "config_sha256": _config_digest(cfg),
        "source_commit": _git_commit(),
        "prediction_count": dataset.metadata["prediction_count"],
        "worldcup_count": dataset.metadata["worldcup_count"],
        "methods": list(results),
        "results_dir": "results",
        "images_dir": "images",
        "artifacts": [
            "results/REPORT.md", "results/DRAFT.md", "results/summary.json",
            "results/comparison_config.json", "results/run_config.json",
            "results/dataset_metadata.json",
            "results/performance_metrics.csv", "results/logz_history.csv",
            "images/logz_overlay_train_test.png",
            "results/ekf_predictions.json", "results/sqmc_predictions.json",
            "results/ekf_metrics.json", "results/sqmc_metrics.json",
            "results/ekf/fitted_params.json", "results/sqmc/fitted_params.json",
            "results/sqmc_prediction_diagnostics.json", "results/sqmc_final_fixture.npz",
            "images/ekf_top5_strengths.png", "images/sqmc_top5_strengths.png",
            "images/ekf_pre_worldcup_rankings.png", "images/sqmc_pre_worldcup_rankings.png",
            "images/ekf_post_worldcup_rankings.png", "images/sqmc_post_worldcup_rankings.png",
        ],
    })
    from rbsqmc.comparison.sqmc_ekf.scripts.validate_sqmc_ekf_outputs import validate_artifacts
    validate_artifacts(run_dir, cfg)
    print(f"Comparison complete: {run_dir}", flush=True)


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _unflatten_metrics(flat):
    """Rebuild a Metrics object from its flattened ``{"all": ..., "worldcup": ...}`` dict."""
    return eval_mod.Metrics(all=flat["all"], worldcup=flat["worldcup"])


# Execution-specific fields that legitimately differ between a local EKF run
# and a GPU SQMC run; every other field must match for a fair comparison.
_EXECUTION_FIELDS = frozenset({
    "gpu", "gpu_type", "colab_timeout", "setup_timeout", "transfer_timeout",
    "session", "session_name", "run_id", "resolved_utc", "source_bundle_sha256",
})


def _check_partial(name, src, cfg, dataset):
    """Verify a partial run's config and dataset match the combine request."""
    src_results = Path(src) / "results"
    partial_cfg = _read_json(src_results / "comparison_config.json")
    differing = ((set(partial_cfg) ^ set(cfg)) - _EXECUTION_FIELDS) | {
        k for k in set(partial_cfg) & set(cfg)
        if partial_cfg[k] != cfg[k] and k not in _EXECUTION_FIELDS
    }
    if differing:
        raise ValueError(f"{name} run config differs in scientific fields: {sorted(differing)}")
    metadata = _read_json(src_results / "dataset_metadata.json")
    for field in ("source_sha256", "train_count", "test_count", "prediction_count", "worldcup_count"):
        if metadata.get(field) != dataset.metadata.get(field):
            raise ValueError(f"{name} run used a different dataset ({field} mismatch)")
    return metadata


def combine(ekf_dir, sqmc_dir, output_dir, data_path, cfg, smoke=False):
    """Merge a local EKF run and a GPU SQMC run into a complete comparison.

    Reads each partial run's persisted per-method artifacts, copies both sets
    of images, and writes the combined report/plots/CSVs/metadata, then
    validates the merged run. Both partial runs must share the same scientific
    configuration and the same frozen dataset.
    """
    dataset = data_mod_load_dataset(data_path, cfg, smoke=smoke)
    results = {"cfg": cfg}
    metadata = None
    for name, src in (("ekf", ekf_dir), ("sqmc", sqmc_dir)):
        partial_metadata = _check_partial(name, src, cfg, dataset)
        if metadata is not None and partial_metadata != metadata:
            raise ValueError("The two partial runs used different datasets")
        metadata = partial_metadata
        src = Path(src)
        src_results = src / "results"
        summary = _read_json(src_results / name / "summary.json")
        results[name] = {
            "history": _read_json(src_results / f"{name}_history.json"),
            "summary": summary,
            "records": _read_json(src_results / f"{name}_predictions.json"),
            "metrics": _unflatten_metrics(_read_json(src_results / f"{name}_metrics.json")),
            "prediction_sec": summary["prediction_sec"],
        }
    run_dir = Path(output_dir)
    results_dir = run_dir / "results"
    images_dir = run_dir / "images"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    # Copy both partial runs' images (the overlay plot is regenerated below).
    for src in (Path(ekf_dir), Path(sqmc_dir)):
        for path in (src / "images").rglob("*"):
            if path.is_file():
                dest = images_dir / path.relative_to(src / "images")
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)
    # The validator reads each method's summary from results/<method>/; each
    # partial run only contains its own method's summary.
    for name, src in (("ekf", ekf_dir), ("sqmc", sqmc_dir)):
        dest = results_dir / name / "summary.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(src) / "results" / name / "summary.json", dest)
        # Carry the reproducibility checkpoint into the combined run.
        fitted = Path(src) / "results" / name / "fitted_params.json"
        if fitted.is_file():
            shutil.copyfile(fitted, results_dir / name / "fitted_params.json")
        # Carry the final scalar-params export into the combined run.
        scalars = Path(src) / "results" / name / "final_scalar_params.json"
        if scalars.is_file():
            shutil.copyfile(scalars, results_dir / name / "final_scalar_params.json")
    # Carry the SQMC diagnostics and final-fixture NPZ into the combined run.
    for artifact in ("sqmc_prediction_diagnostics.json", "sqmc_final_fixture.npz"):
        src_artifact = Path(sqmc_dir) / "results" / artifact
        if src_artifact.is_file():
            shutil.copyfile(src_artifact, results_dir / artifact)
    _save_json(results_dir / "comparison_config.json", cfg)
    _save_json(results_dir / "run_config.json", compute_setup(cfg))
    # Retain the actual training machines; the combined run itself runs on CPU.
    for name, src in (("ekf", ekf_dir), ("sqmc", sqmc_dir)):
        shutil.copyfile(Path(src) / "results" / "run_config.json",
                        results_dir / name / "run_config.json")
    _save_json(results_dir / "dataset_metadata.json", dataset.metadata)
    # Re-persist the per-method artifacts at the combined run's top level.
    for name in ("ekf", "sqmc"):
        _save_json(results_dir / f"{name}_predictions.json", results[name]["records"])
        _save_json(results_dir / f"{name}_metrics.json", _metrics_flat(results[name]["metrics"]))
    _write_combined(results_dir, images_dir, dataset, results, cfg, run_dir)
    return results, str(run_dir)


def write_performance_metrics_csv(run_dir, results):
    """Write one row per method with compute time and prediction metrics."""
    rows = []
    for method in ("ekf", "sqmc"):
        r = results[method]
        m = r["metrics"].worldcup or r["metrics"].all
        s = r["summary"]
        rows.append({
            "method": method,
            "compile_sec": s["compilation_sec"],
            "train_execution_sec": s["execution_sec"],
            "prediction_sec": r["prediction_sec"],
            "pipeline_sec": s["compilation_sec"] + s["execution_sec"] + r["prediction_sec"],
            "final_train_logz": s["final_train_logz"],
            "final_test_logz": s["final_test_logz"],
            "best_test_logz": s["best_test_logz"],
            "mean_brier_score": m.get("mean_brier_score"),
            "brier_skill_score_vs_uniform": m.get("brier_skill_score_vs_uniform"),
            "outcome_accuracy": m.get("outcome_accuracy"),
            "exact_score_accuracy": m.get("exact_score_accuracy"),
            "mean_log_likelihood": m.get("mean_log_likelihood"),
            "n_scored": m.get("n_scored"),
        })
    path = os.path.join(run_dir, "performance_metrics.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_logz_history_csv(run_dir, results):
    """Write per-epoch train/test logZ for both methods (long format)."""
    rows = []
    for method in ("ekf", "sqmc"):
        for h in results[method]["history"]:
            rows.append({"method": method, "epoch": h["epoch"],
                         "train_logz": h["train_logz"], "test_logz": h["test_logz"]})
    path = os.path.join(run_dir, "logz_history.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "epoch", "train_logz", "test_logz"])
        writer.writeheader()
        writer.writerows(rows)


def _run_method(method, methods, dataset, cfg, root, results_dir, images_dir,
                saved_checkpoint=None):
    """Train or replay a method, then predict and evaluate it."""
    if saved_checkpoint is None:
        raw, history, summary = methods.train(method, os.path.join(results_dir, method))
        params = methods.params(method, raw)
    else:
        saved_checkpoint = Path(saved_checkpoint)
        raw = None
        params = train.load_fitted_params(saved_checkpoint, method)
        source_results = saved_checkpoint.parent.parent
        history_path = source_results / f"{method}_history.json"
        if not history_path.is_file():
            # A combined bundle carries the fitted checkpoint under
            # ``combined/results/<method>/`` but keeps per-method histories in
            # the original partial run under ``<run>/<method>/results/``.
            history_path = (
                source_results.parent.parent / method / "results"
                / f"{method}_history.json"
            )
        history = _read_json(history_path)
        summary = dict(_read_json(saved_checkpoint.parent / "summary.json"))
        summary["evaluation_only"] = True
        summary["parameter_source"] = str(saved_checkpoint)

    # Time the prediction + evaluation phase separately from training.
    pred_start = time.perf_counter()
    if method == "sqmc":
        key = jax.random.fold_in(root, 2_000_000)
        scales = sqmc_match_scales(
            dataset.inputs.friendly, params["friendly_scale"],
            cfg.get("match_scale", 1.0),
        )
        result, augmented = run_filter_sqmc(
            key, dataset.sqmc, params["model"],
            cfg["n_particles"], cfg["max_goals"],
            match_scales=scales,
        )
        pred = predict_mod.predict_sqmc_from_history(
            dataset, params, cfg, result, scales
        )
        states = plots_mod.wrap_sqmc(result, len(dataset.teams))
        plots_mod.plot_correlation(
            params["model"], augmented, dataset.teams, images_dir
        )
        # Persist per-forecast diagnostics and the final-fixture NPZ so the
        # Spain–Argentina reversal is reproducible and explainable. The scope
        # config controls how many forecasts are written (all / worldcup /
        # final); the final-fixture NPZ is always written. The diagnostics are
        # bound to the exact fitted-params checkpoint via its SHA-256.
        checkpoint_path = os.path.join(results_dir, "sqmc", "fitted_params.json")
        checkpoint_hash = _file_sha256(checkpoint_path)
        payload, final_npz = diag_mod.build_diagnostics(
            dataset, params, cfg, result, scales,
            source_revision=_git_commit(),
            dataset_hash=dataset.metadata.get("source_sha256"),
            filter_key_provenance="fold_in(root, 2_000_000)",
            schema_version=1,
            scope=cfg.get("diagnostics_scope", "worldcup"),
            checkpoint_hash=checkpoint_hash,
        )
        diag_mod.write_diagnostics(results_dir, payload, final_npz)
    else:
        pred = predict_mod.predict_ekf(dataset, params, cfg)
        states, augmented = _ekf_states(
            dataset, params, len(dataset.teams), cfg.get("match_scale", 1.0)
        )

    records = eval_mod.build_records(dataset, pred.grids, pred.logp)
    metrics = eval_mod.compute_metrics(records)
    prediction_sec = time.perf_counter() - pred_start

    plots_mod.plot_ranking_trajectory(
        states, dataset.teams, dataset.sqmc.timestamp, images_dir, method,
        pre_index=dataset.train_count + dataset.test_count,
        post_index=-1,
    )

    _write_final_scalar_params(
        results_dir, method, params, summary, cfg, dataset,
        checkpoint_hash=_file_sha256(os.path.join(results_dir, method, "fitted_params.json")),
    )

    return {"raw": raw, "params": params, "history": history, "summary": summary,
            "pred": pred, "records": records, "metrics": metrics,
            "states": states, "prediction_sec": prediction_sec}


def _write_final_scalar_params(results_dir, method, params, summary, cfg, dataset, checkpoint_hash):
    """Persist the four final fitted scalar parameters for one method.

    Uses the constrained final-epoch parameters (the same epoch used for
    evaluation), excluding means and covariance quantities. The configured
    non-friendly ``match_scale`` is recorded as run metadata, not as an
    estimated parameter.
    """
    if method == "ekf":
        scalars = {
            "alpha": float(params["alpha"]),
            "beta": float(params["beta"]),
            "kappa": float(params["kappa"]),
            "friendly_scale": float(params["friendly_scale"]),
        }
    else:
        scalars = {
            "alpha": float(params["model"].alpha),
            "beta": float(params["model"].beta),
            "kappa": float(params["model"].kappa),
            "friendly_scale": float(params["friendly_scale"]),
        }
    payload = {
        "method": method,
        "checkpoint_policy": summary["checkpoint_policy"],
        "checkpoint_epoch": summary["n_epochs_completed"],
        "run_id": cfg.get("run_id"),
        "source_revision": _git_commit(),
        "dataset_hash": dataset.metadata.get("source_sha256"),
        "checkpoint_hash": checkpoint_hash,
        "match_scale": cfg.get("match_scale", 1.0),
        "parameters": scalars,
    }
    _save_json(os.path.join(results_dir, method, "final_scalar_params.json"), payload)
    return payload


_SCALAR_MEANINGS = {
    "alpha": ("Baseline log scoring rate", "log goals"),
    "beta": ("Log rate of the shared Poisson component", "log goals"),
    "kappa": ("OU mean-reversion rate", "per day"),
    "friendly_scale": ("Strength scaling for friendly matches", "dimensionless"),
}


def _write_scalar_comparison(results_dir, cfg, run_dir):
    """Read both per-method scalar exports and write the combined comparison
    CSV/JSON plus a generated LaTeX table."""
    ekf = _read_json(os.path.join(results_dir, "ekf", "final_scalar_params.json"))
    sqmc = _read_json(os.path.join(results_dir, "sqmc", "final_scalar_params.json"))
    rows = []
    for param in ("alpha", "beta", "kappa", "friendly_scale"):
        meaning, unit = _SCALAR_MEANINGS[param]
        e = ekf["parameters"][param]
        s = sqmc["parameters"][param]
        rows.append({
            "parameter": param,
            "meaning": meaning,
            "unit": unit,
            "ekf": e,
            "sqmc": s,
            "sqmc_minus_ekf": s - e,
        })
    # Derived rows (not additional learned parameters).
    rows.append({"parameter": "exp(alpha)", "meaning": "Baseline scoring rate",
                 "unit": "goals", "ekf": math.exp(ekf["parameters"]["alpha"]),
                 "sqmc": math.exp(sqmc["parameters"]["alpha"]),
                 "sqmc_minus_ekf": math.exp(sqmc["parameters"]["alpha"]) - math.exp(ekf["parameters"]["alpha"])})
    rows.append({"parameter": "exp(beta)", "meaning": "Shared-component rate",
                 "unit": "goals", "ekf": math.exp(ekf["parameters"]["beta"]),
                 "sqmc": math.exp(sqmc["parameters"]["beta"]),
                 "sqmc_minus_ekf": math.exp(sqmc["parameters"]["beta"]) - math.exp(ekf["parameters"]["beta"])})
    rows.append({"parameter": "log(2)/kappa", "meaning": "OU half-life",
                 "unit": "days", "ekf": math.log(2) / ekf["parameters"]["kappa"],
                 "sqmc": math.log(2) / sqmc["parameters"]["kappa"],
                 "sqmc_minus_ekf": math.log(2) / sqmc["parameters"]["kappa"] - math.log(2) / ekf["parameters"]["kappa"]})
    csv_path = os.path.join(results_dir, "final_scalar_params_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["parameter", "meaning", "unit", "ekf", "sqmc", "sqmc_minus_ekf"])
        writer.writeheader()
        writer.writerows(rows)
    _save_json(os.path.join(results_dir, "final_scalar_params_comparison.json"), {
        "run_id": os.path.basename(run_dir),
        "checkpoint_epoch_ekf": ekf["checkpoint_epoch"],
        "checkpoint_epoch_sqmc": sqmc["checkpoint_epoch"],
        "rows": rows,
    })
    _write_scalar_latex(results_dir, rows, ekf, sqmc, run_dir)


def _write_scalar_latex(results_dir, rows, ekf, sqmc, run_dir):
    """Generate a LaTeX table from the scalar comparison artifact.

    Columns: Parameter / Interpretation / EKF / RB-SQMC. Parameter names are
    rendered as safe labels and the run ID is escaped so the table compiles
    even when the run ID contains underscores or other special characters.
    """
    run_id = _latex_escape(os.path.basename(run_dir))
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Final fitted scalar parameters (run \\texttt{%s}, final epoch EKF %d / SQMC %d).}"
        % (run_id, ekf["checkpoint_epoch"], sqmc["checkpoint_epoch"]),
        "\\begin{tabular}{lccc}",
    ]
    lines += _scalar_table_body(rows)
    lines += ["\\end{tabular}", "\\end{table}"]
    with open(os.path.join(results_dir, "final_scalar_params_table.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def _sqmc_states(dataset, params, cfg, root):
    """Weighted SQMC posterior moments plus the augmented gamma trajectory.

    Deprecated: the evaluation filter is now run once in ``_run_method`` and
    its history is shared by predictions and rankings. This wrapper is retained
    only for callers that still need a standalone filter; it reruns the full
    filter with a distinct key and must not be used for the main comparison.
    """
    key = jax.random.fold_in(root, 3_000_000)
    scales = sqmc_match_scales(
        dataset.inputs.friendly, params["friendly_scale"],
        cfg.get("match_scale", 1.0),
    )
    result, augmented = run_filter_sqmc(
        key, dataset.sqmc, params["model"], cfg["n_particles"], cfg["max_goals"],
        match_scales=scales,
    )
    return plots_mod.wrap_sqmc(result, len(dataset.teams)), augmented


def _ekf_states(dataset, params, num_teams, match_scale=1.0):
    """Native EKF moments wrapped as a single-particle FilterStates."""
    from rbsqmc.src.model.ekf import model as ekf

    history = ekf.run_filter(dataset.inputs, params, num_teams, match_scale=match_scale)
    mean, cov = ekf.synchronized_moments(
        dataset.inputs, history, params, num_teams, match_scale=match_scale
    )
    # synchronized_moments returns post-match states only. Restore the prior
    # so both methods' index i means "before match i", including the WC split.
    mean = np.concatenate([np.asarray(history["mean"][:1]), np.asarray(mean)])
    cov = np.concatenate([np.asarray(history["cov"][:1]), np.asarray(cov)])
    return plots_mod.MeanFilterStates(mean, cov), None


def main():
    parser = argparse.ArgumentParser(description="Run SQMC vs EKF comparison")
    parser.add_argument("--config", required=True, type=str,
                        help="Path to the comparison config JSON.")
    parser.add_argument("--data", default=DEFAULT_DATA, type=str,
                        help="Path to the results CSV/parquet.")
    parser.add_argument("--smoke", action="store_true",
                        help="Use a smoke subset for a fast run.")
    parser.add_argument("--output-dir", default=None, type=str,
                        help="Optional explicit output directory.")
    parser.add_argument("--methods", default="both", choices=["both", "ekf", "sqmc"],
                        help="Which methods to run (default: both).")
    parser.add_argument("--evaluate-params", default=None, type=str,
                        help="Replay a saved fitted_params.json without training; currently SQMC only.")
    parser.add_argument("--combine", nargs=2, metavar=("EKF_DIR", "SQMC_DIR"),
                        help="Merge a local EKF run and a GPU SQMC run into a complete comparison.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    if args.smoke:
        cfg["smoke"] = True
        cfg["n_epochs"] = min(cfg.get("n_epochs", 100), 3)
        cfg["n_reps"] = min(cfg.get("n_reps", 25), 2)
        cfg["n_particles"] = min(cfg.get("n_particles", 512), 64)

    if args.combine:
        if args.methods != "both" or args.smoke:
            parser.error("--combine cannot be combined with --methods or --smoke")
        ekf_dir, sqmc_dir = args.combine
        if not args.output_dir:
            parser.error("--combine requires --output-dir")
        # A stored ``smoke`` flag in the config is authoritative: partial runs
        # trained on the smoke subset whenever their config recorded it, so the
        # combine step must load the same subset regardless of this CLI flag.
        combine(ekf_dir, sqmc_dir, args.output_dir, args.data, cfg,
                smoke=bool(cfg.get("smoke")))
        return

    methods = ("ekf", "sqmc") if args.methods == "both" else (args.methods,)
    if args.evaluate_params and args.methods != "sqmc":
        parser.error("--evaluate-params requires --methods sqmc")
    run(cfg, args.data, smoke=args.smoke, output_dir=args.output_dir,
        methods=methods, evaluate_params=args.evaluate_params)


if __name__ == "__main__":
    main()

"""Validate comparison contents before completion and after archive download.

Standard-library only so the local Colab launcher needs no scientific packages.
Accept an exact run directory, or select the chronologically latest timestamped
run under an outputs directory when invoked as a CLI.
"""

import csv
from datetime import date, datetime
import hashlib
import json
import math
import re
from pathlib import Path
import struct
import sys
import zlib

import jax.numpy as jnp
import numpy as np


METHODS = ("ekf", "sqmc")
# Bump when the repaired output schema changes so old runs (which lack the
# diagnostics artifacts) are identified as pre-repair rather than mistaken for
# complete repaired runs.
SCHEMA_VERSION = 1
REQUIRED_RESULTS = (
    "comparison_config.json", "run_config.json", "dataset_metadata.json",
    "run_metadata.json", "summary.json", "performance_metrics.csv",
    "logz_history.csv", "REPORT.md", "DRAFT.md",
)
REQUIRED_METHOD_IMAGES = (
    "top5_strengths.png", "timeseries_states.png",
    "pre_worldcup_rankings.png", "post_worldcup_rankings.png",
)
FIXTURE_FIELDS = ("date", "home", "away", "actual_home_score", "actual_away_score",
                  "tournament", "worldcup_eligible")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _nonempty(path):
    _require(path.is_file() and path.stat().st_size > 0, f"Missing or empty artifact: {path}")


def _ensure_finite(label, tree):
    if isinstance(tree, dict):
        for value in tree.values():
            _ensure_finite(label, value)
    elif isinstance(tree, list):
        for value in tree:
            _ensure_finite(label, value)
    elif isinstance(tree, (int, float)):
        _require(math.isfinite(tree), f"{label}: non-finite value {tree}")


def _json(path):
    _nonempty(path)
    value = json.loads(path.read_text())
    _ensure_finite(str(path), value)
    return value


def _csv(path):
    _nonempty(path)
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    _require(bool(rows), f"Empty CSV: {path}")
    return rows


def _close(actual, expected, label):
    if expected is None:
        _require(actual is None or actual == "", f"{label}: expected an unscored value")
    elif isinstance(expected, (int, float)):
        _require(not isinstance(actual, bool), f"{label}: expected a number")
        value = float(actual)
        _require(math.isfinite(value) and math.isclose(value, expected, rel_tol=1e-6, abs_tol=1e-8),
                 f"{label}: {actual} != {expected}")
    else:
        _require(actual == expected, f"{label}: {actual!r} != {expected!r}")


def _png(path):
    """Check PNG structure, chunk checksums and the compressed image stream."""
    _nonempty(path)
    data = path.read_bytes()
    _require(data[:8] == b"\x89PNG\r\n\x1a\n", f"Invalid PNG: {path}")
    offset, chunks, compressed = 8, [], bytearray()
    while offset < len(data):
        _require(offset + 12 <= len(data), f"Truncated PNG: {path}")
        size = struct.unpack_from(">I", data, offset)[0]
        end = offset + 12 + size
        _require(end <= len(data), f"Truncated PNG chunk: {path}")
        kind, content = data[offset + 4:offset + 8], data[offset + 8:end - 4]
        crc = struct.unpack_from(">I", data, end - 4)[0]
        _require(zlib.crc32(kind + content) == crc, f"PNG checksum mismatch: {path}")
        if not chunks:
            _require(kind == b"IHDR" and size == 13, f"Invalid PNG header: {path}")
            width, height = struct.unpack_from(">II", content)
            _require(width > 0 and height > 0, f"Empty PNG dimensions: {path}")
        if kind == b"IDAT":
            compressed.extend(content)
        chunks.append(kind)
        offset = end
        if kind == b"IEND":
            _require(size == 0 and end == len(data), f"Invalid PNG ending: {path}")
            break
    _require(chunks[-1:] == [b"IEND"] and compressed, f"Missing PNG image data: {path}")
    decoder = zlib.decompressobj()
    pixels = decoder.decompress(compressed)
    _require(bool(pixels) and decoder.eof and not decoder.unused_data, f"Invalid PNG stream: {path}")


def _predictions(records, cfg, count):
    _require(isinstance(records, list) and len(records) == count, "Prediction count mismatch")
    size = cfg["max_goals"] + 1
    previous_date = date.fromisoformat(cfg["prediction_start_date"])
    fixtures = []
    for index, record in enumerate(records):
        label = f"prediction {index}"
        when = date.fromisoformat(record["date"])
        _require(when >= previous_date, f"{label}: wrong split or chronological order")
        previous_date = when
        _require(record["home"] and record["away"] and record["home"] != record["away"],
                 f"{label}: invalid teams")
        eligible = when.year == 2026 and record["tournament"] == "FIFA World Cup"
        _require(record["worldcup_eligible"] is eligible, f"{label}: wrong World Cup eligibility")
        cells = record["score_probabilities"]
        _require(len(cells) == size**2, f"{label}: incomplete score grid")
        grid = {}
        for cell in cells:
            h, a, probability = cell["home"], cell["away"], cell["probability"]
            _require(type(h) is int and type(a) is int and 0 <= h < size and 0 <= a < size,
                     f"{label}: invalid grid coordinates")
            _require((h, a) not in grid, f"{label}: duplicate grid coordinates")
            _require(type(probability) in (int, float) and 0 <= probability <= 1,
                     f"{label}: probability outside [0, 1]")
            grid[h, a] = probability
        _close(sum(grid.values()), 1., f"{label} grid mass")
        # Verify outcome probabilities and score selection against the saved grid,
        # rather than trusting mutually inconsistent derived fields.
        probs = [sum(p for (h, a), p in grid.items() if h > a),
                 sum(p for (h, a), p in grid.items() if h == a),
                 sum(p for (h, a), p in grid.items() if h < a)]
        for field, value in zip(("prob_home_win", "prob_draw", "prob_away_win"), probs):
            _close(record[field], value, f"{label} {field}")
        best = max(range(size**2), key=lambda i: grid[divmod(i, size)])
        _require((record["predicted_home_score"], record["predicted_away_score"]) == divmod(best, size),
                 f"{label}: predicted score disagrees with grid")
        h, a = record["actual_home_score"], record["actual_away_score"]
        _require(type(h) is int and type(a) is int and -1 <= h < size and -1 <= a < size,
                 f"{label}: invalid actual scores")
        # The existing predictors use log(p + 1e-12); validate that convention
        # without changing the scoring model as part of an artifact check.
        expected_logp = math.log(grid[h, a] + 1e-12) if h >= 0 and a >= 0 else 0.
        _close(record["log_likelihood"], expected_logp, f"{label} log probability")
        fixtures.append(tuple(record[field] for field in FIXTURE_FIELDS))
    return fixtures


def _metrics(records):
    known = [r for r in records if min(r["actual_home_score"], r["actual_away_score"]) >= 0]
    brier, exact, outcome, logp = [], [], [], []
    for record in known:
        h, a = record["actual_home_score"], record["actual_away_score"]
        actual = 0 if h > a else 1 if h == a else 2
        probs = [record["prob_home_win"], record["prob_draw"], record["prob_away_win"]]
        brier.append(sum((p - int(i == actual))**2 for i, p in enumerate(probs)))
        outcome.append(max(range(3), key=probs.__getitem__) == actual)
        exact.append((record["predicted_home_score"], record["predicted_away_score"]) == (h, a))
        logp.append(record["log_likelihood"])
    n = len(known)
    mean_brier = sum(brier) / n if n else None
    return dict(n_predictions=len(records), n_scored=n, brier_score_definition="sum_over_home_draw_away",
                mean_brier_score=mean_brier, uniform_reference_brier_score=2 / 3,
                brier_skill_score_vs_uniform=1 - mean_brier / (2 / 3) if n else None,
                total_log_likelihood=sum(logp) if n else None,
                mean_log_likelihood=sum(logp) / n if n else None,
                exact_score_accuracy=sum(exact) / n if n else None,
                outcome_accuracy=sum(outcome) / n if n else None)


def validate_partial(run_dir, config, method):
    """Check a single machine's artifacts before collecting or reusing them.

    Cross-method fixture agreement and the combined reports are checked by
    validate_artifacts after combining; each partial must already be complete.
    """
    root = Path(run_dir)
    results = root / "results"
    cfg = _json(results / "comparison_config.json")
    _require(cfg == config, "Effective comparison configuration mismatch")
    provenance = _json(results / "run_config.json")
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    _require(provenance["config_sha256"] == digest, "Partial config digest mismatch")
    if "source_commit" in cfg:
        _require(provenance["source_commit"] == cfg["source_commit"], "Partial source commit mismatch")
    summary = _json(results / method / "summary.json")
    history = _json(results / f"{method}_history.json")
    _require(summary["n_epochs_completed"] == cfg["n_epochs"], f"{method}: incomplete training")
    _require([row["epoch"] for row in history] == list(range(1, cfg["n_epochs"] + 1)),
             f"{method}: missing, duplicated or unordered epochs")
    for field in ("train_logz", "test_logz"):
        _close(summary[f"final_{field}"], history[-1][field], f"{method} final {field}")
    metadata = _json(results / "dataset_metadata.json")
    records = _json(results / f"{method}_predictions.json")
    _predictions(records, cfg, metadata["prediction_count"])
    metrics = _json(results / f"{method}_metrics.json")
    wc = [r for r in records if r["worldcup_eligible"]]
    _require(len(wc) == metadata["worldcup_count"], f"{method}: World Cup count mismatch")
    scored_wc = [r for r in wc if min(r["actual_home_score"], r["actual_away_score"]) >= 0]
    for group, expected in dict(all=_metrics(records), worldcup=_metrics(scored_wc) if scored_wc else None).items():
        if expected is None:
            _require(metrics[group] is None, f"{method}: metrics for unscored fixtures")
        else:
            for field, value in expected.items():
                _close(metrics[group][field], value, f"{method} {group} {field}")
    if method == "sqmc":
        _validate_checkpoint(results, "sqmc", cfg)
        _validate_sqmc_diagnostics(results, metadata, cfg, records=records)
    else:
        _validate_checkpoint(results, "ekf", cfg)
    # Each partial must persist and validate its own scalar export.
    _validate_scalar_params(results, cfg, metadata=metadata, methods=(method,))
    for suffix in REQUIRED_METHOD_IMAGES:
        _png(root / "images" / f"{method}_{suffix}")


def _validate_checkpoint(results, method, cfg):
    """Validate one method's fitted-params checkpoint.

    Separate from SQMC diagnostics so a partial run validates only its own
    method's checkpoint (a SQMC-only run must not require the EKF checkpoint,
    and vice versa). Validates the parameter schema and reconstructs the
    constrained parameters from the raw parameters to check consistency.
    """
    fitted = _json(results / method / "fitted_params.json")
    _require(fitted.get("method") == method, f"{method}: fitted-params method mismatch")
    _require(fitted.get("checkpoint_policy") == "final_epoch", f"{method}: wrong checkpoint policy")
    _require(fitted.get("checkpoint_epoch") == cfg["n_epochs"], f"{method}: wrong checkpoint epoch")
    _require(fitted.get("team_id_to_name"), f"{method}: missing team mapping")
    raw = fitted.get("raw")
    constrained = fitted.get("constrained")
    _require(isinstance(raw, dict) and raw, f"{method}: checkpoint missing raw parameters")
    _require(isinstance(constrained, dict) and constrained, f"{method}: checkpoint missing constrained parameters")
    if method == "ekf":
        _require(all(k in raw for k in ("init_sd", "init_corr", "kappa", "alpha", "beta", "friendly_scale")),
                 "ekf: raw parameters missing required fields")
        _require(all(k in constrained for k in ("init_mean", "init_cov", "init_chol_cov", "kappa", "alpha", "beta", "friendly_scale")),
                 "ekf: constrained parameters missing required fields")
        # Reconstruct constrained from raw and compare every field, including
        # the covariance/mean matrices and their dimensions.
        from rbsqmc.src.model.ekf import model as ekf
        rebuilt = ekf.constrain(raw)
        for field in ("alpha", "beta", "kappa", "friendly_scale"):
            _close(constrained[field], float(rebuilt[field]), f"ekf constrained {field} vs raw")
        for field in ("init_mean", "init_cov", "init_chol_cov"):
            _require(np.asarray(constrained[field]).shape == np.asarray(rebuilt[field]).shape,
                     f"ekf constrained {field} shape mismatch")
            # Cross-platform reconstruction: the saved `constrained` was
            # computed on the GPU worker, but this validator re-decodes from
            # `raw` on the local CPU. Cholesky round-trips differ by ~1e-8
            # absolute / ~1e-6 relative purely from platform float32 noise, so
            # use a tolerance that still catches real corruption but tolerates
            # that noise (see the SQMC block below).
            np.testing.assert_allclose(
                np.asarray(constrained[field]), np.asarray(rebuilt[field]),
                rtol=1e-4, atol=1e-6,
            )
    else:
        _require(isinstance(raw.get("model"), dict) and raw.get("model"),
                 "sqmc: raw model parameters missing")
        _require(all(k in raw["model"] for k in ("gamma_0_chol", "b_chol_raw", "kappa_raw", "alpha", "beta")),
                 "sqmc: raw model parameters missing required fields")
        _require(isinstance(constrained.get("model"), dict) and constrained.get("model"),
                 "sqmc: constrained model parameters missing")
        _require(all(k in constrained["model"] for k in ("mean_0", "gamma_0", "B", "kappa", "alpha", "beta")),
                 "sqmc: constrained model parameters missing required fields")
        _require("friendly_scale" in constrained, "sqmc: constrained friendly_scale missing")
        # Reconstruct constrained from raw and the saved mean, then compare
        # every field including mean_0, gamma_0, and B with their dimensions.
        from rbsqmc.src.model.ekf import model as ekf
        from rbsqmc.src.utils.helpers import decode_EM_params
        from rbsqmc.src.utils.type import RawEMParams
        saved_mean = constrained["model"]["mean_0"]
        raw_model = RawEMParams(**{k: jnp.asarray(v) for k, v in raw["model"].items()})
        rebuilt_model = decode_EM_params(raw_model, jnp.asarray(saved_mean))
        _close(constrained["model"]["alpha"], float(rebuilt_model.alpha), "sqmc constrained alpha vs raw")
        _close(constrained["model"]["beta"], float(rebuilt_model.beta), "sqmc constrained beta vs raw")
        _close(constrained["model"]["kappa"], float(rebuilt_model.kappa), "sqmc constrained kappa vs raw")
        _close(constrained["friendly_scale"], float(ekf.positive(jnp.asarray(raw["friendly_scale"]))),
               "sqmc constrained friendly_scale vs raw")
        for field in ("mean_0", "gamma_0", "B"):
            _require(np.asarray(constrained["model"][field]).shape == np.asarray(getattr(rebuilt_model, field)).shape,
                     f"sqmc constrained {field} shape mismatch")
            # Cross-platform reconstruction: the saved `constrained` was
            # computed on the GPU worker, but this validator re-decodes from
            # `raw` on the local CPU. The Cholesky round-trip (encode on GPU,
            # decode on CPU) differs by ~2e-8 absolute / ~4e-6 relative purely
            # from platform float32 noise, so use a tolerance that still
            # catches real corruption but tolerates that noise.
            np.testing.assert_allclose(
                np.asarray(constrained["model"][field]), np.asarray(getattr(rebuilt_model, field)),
                rtol=1e-4, atol=1e-6,
            )
    return fitted


def _validate_sqmc_diagnostics(results, metadata, cfg, records=None):
    """Validate the repaired SQMC diagnostics and fitted-params artifacts.

    ``records`` (optional) is the SQMC prediction records list; when supplied,
    each diagnostic entry is bound to its prediction record (fixture alignment,
    team IDs/names, indices, and outcome probabilities).
    """
    diag = _json(results / "sqmc_prediction_diagnostics.json")
    _require(diag.get("schema_version") == SCHEMA_VERSION,
             f"Diagnostics schema version mismatch (expected {SCHEMA_VERSION})")
    _require(diag.get("n_particles") == cfg["n_particles"], "Diagnostics particle count mismatch")
    _require(diag.get("max_goals") == cfg["max_goals"], "Diagnostics max_goals mismatch")
    # Bind the diagnostics to the exact fitted-params checkpoint via its hash.
    checkpoint_path = results / "sqmc" / "fitted_params.json"
    _nonempty(checkpoint_path)
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    _require(diag.get("checkpoint_hash") == checkpoint_hash,
             "Diagnostics checkpoint hash does not match the saved fitted params")
    # Bind the diagnostics to the dataset via its hash.
    _require(diag.get("dataset_hash") == metadata.get("source_sha256"),
             "Diagnostics dataset hash does not match the run metadata")
    # Bind source revision and filter-key provenance to the run metadata.
    run_config = _json(results / "run_config.json")
    _require(diag.get("source_revision") == run_config.get("source_commit"),
             "Diagnostics source revision does not match the run config")
    _require(diag.get("filter_key_provenance") == "fold_in(root, 2_000_000)",
             "Diagnostics filter-key provenance does not match the evaluation filter")
    # Team ID mapping comes from the SQMC checkpoint.
    fitted = _json(checkpoint_path)
    team_id_to_name = fitted.get("team_id_to_name", {})
    scope = diag.get("scope", "worldcup")
    _require(scope in ("all", "worldcup", "final"), f"Invalid diagnostics scope: {scope}")
    forecasts = diag["forecasts"]
    # The number of persisted forecasts depends on the scope, not the full
    # prediction count. The final-fixture NPZ is always written, so a run with
    # no 2026 World Cup fixtures in the prediction split legitimately has zero
    # worldcup-scope forecasts.
    _require(len(forecasts) <= metadata["prediction_count"], "Diagnostics forecast count exceeds prediction count")
    if scope in ("all", "final"):
        _require(len(forecasts) >= 1, f"Diagnostics scope {scope} must contain at least one forecast")
    # The diagnostics store the full-history match index t (including the
    # observed prefix). Prediction records contain only the prediction split,
    # so the prediction index is p = t - prefix.
    prefix = metadata["train_count"] + metadata["test_count"]
    # Bind each entry to its prediction record when available.
    if records is not None:
        _require(len(records) == metadata["prediction_count"], "Prediction count mismatch")
        eligible = [r for r in records if r["worldcup_eligible"]]
        if scope == "worldcup":
            _require(len(forecasts) == len(eligible),
                     "Diagnostics worldcup scope must match the eligible prediction records")
        elif scope == "all":
            _require(len(forecasts) == len(records),
                     "Diagnostics all scope must match every prediction record")
        elif scope == "final":
            _require(len(forecasts) == 1, "Diagnostics final scope must contain exactly one forecast")
        # Require the exact ordered prediction indices selected by the scope,
        # rejecting duplicates and omissions even when counts match.
        if scope == "all":
            expected_p = list(range(len(records)))
        elif scope == "worldcup":
            expected_p = [i for i, r in enumerate(records) if r["worldcup_eligible"]]
        else:  # final
            expected_p = [len(records) - 1]
        actual_p = [entry["full_sequence_match_index"] - prefix for entry in forecasts]
        _require(actual_p == expected_p,
                 "Diagnostics scope does not select exactly the expected ordered prediction indices")
    for entry in forecasts:
        for field in ("home_id", "away_id", "full_sequence_match_index", "history_index",
                      "current_scale", "raw_grid_mass", "ess_before_resampling", "ess_after_update"):
            _ensure_finite(f"diagnostics {field}", float(entry[field]))
        _require(float(entry["current_scale"]) > 0, "Diagnostics scale must be positive")
        _require(0.0 < float(entry["raw_grid_mass"]) <= 1.0 + 1e-6,
                 "Diagnostics raw grid mass must be in (0, 1]")
        _require(1.0 - 1e-6 <= float(entry["ess_before_resampling"]) <= float(entry["n_particles"]) + 1e-6,
                 "Diagnostics ESS before resampling out of range")
        _require(1.0 - 1e-6 <= float(entry["ess_after_update"]) <= float(entry["n_particles"]) + 1e-6,
                 "Diagnostics ESS after update out of range")
        for field in ("predictive_means", "posterior_means"):
            for k, v in entry[field].items():
                _ensure_finite(f"diagnostics {field}.{k}", float(v))
        # Validate the total-strength-difference distribution: quantile
        # ordering, fraction_positive in [0, 1], and mean/total consistency.
        tsd = entry["total_strength_difference"]
        for k in ("mean", "q05", "q50", "q95"):
            _ensure_finite(f"diagnostics total_strength_difference.{k}", float(tsd[k]))
        _require(float(tsd["q05"]) <= float(tsd["q50"]) <= float(tsd["q95"]),
                 "Diagnostics total-strength quantiles must be ordered q05 <= q50 <= q95")
        _require(0.0 <= float(tsd["fraction_positive"]) <= 1.0,
                 "Diagnostics fraction_positive must be in [0, 1]")
        # The strength-difference mean must equal the difference of the
        # predictive team totals (attack + defence).
        pm = entry["predictive_means"]
        total_diff = (pm["home_attack"] + pm["home_defence"]) - (pm["away_attack"] + pm["away_defence"])
        _close(tsd["mean"], total_diff, "Diagnostics strength-difference mean vs predictive totals")
        # Per-entry particle count and score bound must match configuration.
        _require(entry["n_particles"] == cfg["n_particles"], "Diagnostics entry particle count mismatch")
        _require(entry["max_goals"] == cfg["max_goals"], "Diagnostics entry max_goals mismatch")
        probs = entry["outcome_probabilities"]
        for k in ("home", "draw", "away"):
            _require(0.0 <= float(probs[k]) <= 1.0, f"Diagnostics outcome probability {k} out of range")
        total = sum(float(probs[k]) for k in ("home", "draw", "away"))
        _require(abs(total - 1.0) < 1e-6, "Diagnostics outcome probabilities do not sum to one")
        # Bind to the prediction record when available. The full-history index
        # t maps to prediction index p = t - prefix.
        if records is not None:
            t = entry["full_sequence_match_index"]
            _require(type(t) is int, "Diagnostics full-sequence index must be an integer")
            _require(entry["history_index"] == t + 1,
                     "Diagnostics history_index must equal full_sequence_match_index + 1")
            p = t - prefix
            _require(0 <= p < len(records),
                     "Diagnostics full-sequence index out of range for the prediction split")
            record = records[p]
            _close(entry["current_scale"], _fixture_scale(record, fitted, cfg),
                   "Diagnostics scale vs fixture")
            _require(record["date"] == entry["date"], "Diagnostics date does not match prediction record")
            _require(record["home"] == entry["home"] and record["away"] == entry["away"],
                     "Diagnostics teams do not match prediction record")
            _require(0 <= entry["home_id"] < len(team_id_to_name),
                     "Diagnostics home_id out of range")
            _require(0 <= entry["away_id"] < len(team_id_to_name),
                     "Diagnostics away_id out of range")
            # Resolve each team ID through the checkpoint mapping and require
            # the corresponding name to match the record; an in-range ID alone
            # is insufficient.
            _require(team_id_to_name.get(str(entry["home_id"])) == record["home"],
                     "Diagnostics home_id does not resolve to the record's home team")
            _require(team_id_to_name.get(str(entry["away_id"])) == record["away"],
                     "Diagnostics away_id does not resolve to the record's away team")
            _close(record["prob_home_win"], probs["home"], "Diagnostics home probability vs prediction")
            _close(record["prob_draw"], probs["draw"], "Diagnostics draw probability vs prediction")
            _close(record["prob_away_win"], probs["away"], "Diagnostics away probability vs prediction")
    final_entry = next((e for e in forecasts if records is not None and
                        e["full_sequence_match_index"] == prefix + len(records) - 1), None)
    expected_final_scale = (_fixture_scale(records[-1], fitted, cfg)
                            if records else None)
    _validate_final_npz(
        results / "sqmc_final_fixture.npz", cfg,
        records=records, team_id_to_name=team_id_to_name,
        final_scale=expected_final_scale, final_diagnostic=final_entry,
        checkpoint_alpha=float(fitted["constrained"]["model"]["alpha"]),
        checkpoint_beta=float(fitted["constrained"]["model"]["beta"]),
    )


def _fixture_scale(record, fitted, cfg):
    # Match data.filter_teams: case-insensitive substring, not exact equality.
    friendly = "friendly" in (record.get("tournament") or "").lower()
    scale = (fitted["constrained"]["friendly_scale"] if friendly
             else cfg.get("match_scale", 1.0))
    _require(math.isfinite(float(scale)) and float(scale) > 0, "Invalid fixture scale")
    return float(scale)


def _validate_final_npz(path, cfg, records=None, team_id_to_name=None, final_scale=None,
                        checkpoint_alpha=None, checkpoint_beta=None, final_diagnostic=None):
    """Open and replay the final-fixture NPZ, checking fields, shapes, finite
    coordinates, valid log weights, and (when records/team mapping are given)
    reconstructing the grid and comparing it to the saved final prediction."""
    _nonempty(path)
    with np.load(path, allow_pickle=False) as data:
        for field in ("particles_x_predictive", "log_weights_predictive",
                      "log_weights_posterior", "home_id", "away_id", "scale",
                      "alpha", "beta", "max_goals"):
            _require(field in data, f"Final NPZ missing field {field}")
        particles = data["particles_x_predictive"]
        _require(particles.ndim == 3 and particles.shape[2] == 2,
                 "Final NPZ particles_x_predictive must be (N, num_teams, 2)")
        n = particles.shape[0]
        _require(n == cfg["n_particles"], "Final NPZ particle count mismatch")
        for field in ("log_weights_predictive", "log_weights_posterior"):
            w = data[field]
            _require(w.shape == (n,), f"Final NPZ {field} must be (N,)")
            _require(np.isfinite(w).any(), f"Final NPZ {field} has no finite weight")
            _require(not np.isnan(w).any() and not (w == np.inf).any(),
                     f"Final NPZ {field} contains NaN or +inf")
        _require(np.isfinite(particles).all(), "Final NPZ particles contain non-finite values")
        _require(np.isfinite(data["scale"]).all() and float(data["scale"]) > 0,
                 "Final NPZ scale must be finite and positive")
        _require(np.isfinite(data["alpha"]).all() and np.isfinite(data["beta"]).all(),
                 "Final NPZ alpha/beta must be finite")
        if checkpoint_alpha is not None:
            _close(float(data["alpha"]), checkpoint_alpha, "Final NPZ alpha vs checkpoint")
        if checkpoint_beta is not None:
            _close(float(data["beta"]), checkpoint_beta, "Final NPZ beta vs checkpoint")
        _require(int(data["max_goals"]) == cfg["max_goals"], "Final NPZ max_goals mismatch")
        home_id, away_id = int(data["home_id"]), int(data["away_id"])
        _require(home_id != away_id, "Final NPZ home and away teams must be distinct")
        if team_id_to_name is not None:
            _require(0 <= home_id < len(team_id_to_name), "Final NPZ home_id out of range")
            _require(0 <= away_id < len(team_id_to_name), "Final NPZ away_id out of range")
        if final_scale is not None:
            _require(abs(float(data["scale"]) - final_scale) < 1e-6,
                     "Final NPZ scale does not match the final fixture scale")
        # Predictive weights must be uniform under the bootstrap-filter
        # contract (the forecast uses pre-likelihood uniform weights).
        lw_pred = data["log_weights_predictive"]
        _require(np.allclose(lw_pred, lw_pred[0], atol=1e-8),
                 "Final NPZ predictive log weights must be uniform")
        # Reconstruct the normalized grid and raw mass from the saved
        # predictive coordinates and uniform predictive weights.
        from rbsqmc.src.model.rbsmc.predict import predict_match_score_with_mass
        import jax.numpy as jnp
        grid, raw_mass = predict_match_score_with_mass(
            jnp.asarray(particles), jnp.zeros(n), home_id, away_id,
            float(data["alpha"]), float(data["beta"]), int(data["max_goals"]),
            float(data["scale"]),
        )
        grid = np.asarray(grid)
        _require(0.0 < float(raw_mass) <= 1.0 + 1e-6, "Final NPZ raw mass out of range")
        if final_diagnostic is not None:
            # The raw mass is a sum over the full score grid. The saved value
            # was computed on the GPU worker; this replay recomputes it on the
            # local CPU, so float32 accumulation noise (~1e-6 relative) is
            # expected. Use a looser tolerance than the exact-value `_close`.
            _require(math.isclose(final_diagnostic["raw_grid_mass"], float(raw_mass),
                                  rel_tol=1e-4, abs_tol=1e-6),
                     f"Final diagnostic raw mass vs NPZ replay: "
                     f"{final_diagnostic['raw_grid_mass']} != {float(raw_mass)}")
            _close(final_diagnostic["current_scale"], float(data["scale"]),
                   "Final diagnostic scale vs NPZ")
            posterior_lw = data["log_weights_posterior"]
            posterior_w = np.exp(posterior_lw - np.max(posterior_lw))
            posterior_w /= posterior_w.sum()
            for label, weights in (("predictive_means", np.full(n, 1.0 / n)),
                                   ("posterior_means", posterior_w)):
                for side, team_id in (("home", home_id), ("away", away_id)):
                    means = np.sum(particles[:, team_id, :] * weights[:, None], axis=0)
                    for component, value in zip(("attack", "defence"), means):
                        _close(final_diagnostic[label][f"{side}_{component}"], float(value),
                               f"Final diagnostic {label} {side}_{component} vs NPZ replay")
            _close(final_diagnostic["ess_after_update"], float(1 / np.sum(posterior_w**2)),
                   "Final diagnostic posterior ESS vs NPZ replay")
        # Compare the replayed grid to the saved final prediction record,
        # cell by cell (full-grid replay), and resolve the NPZ team IDs to the
        # final record's names.
        if records is not None:
            final_record = records[-1]
            G = grid.shape[0]
            ii, jj = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
            replayed = {
                "home": float(grid[ii > jj].sum()),
                "draw": float(grid[ii == jj].sum()),
                "away": float(grid[ii < jj].sum()),
            }
            _close(final_record["prob_home_win"], replayed["home"], "Final NPZ home probability vs prediction")
            _close(final_record["prob_draw"], replayed["draw"], "Final NPZ draw probability vs prediction")
            _close(final_record["prob_away_win"], replayed["away"], "Final NPZ away probability vs prediction")
            if team_id_to_name is not None:
                _require(team_id_to_name.get(str(home_id)) == final_record["home"],
                         "Final NPZ home_id does not resolve to the final record's home team")
                _require(team_id_to_name.get(str(away_id)) == final_record["away"],
                         "Final NPZ away_id does not resolve to the final record's away team")
            # Full-grid replay: every cell must match the saved score grid.
            saved_cells = final_record["score_probabilities"]
            for cell in saved_cells:
                h, a = cell["home"], cell["away"]
                _close(cell["probability"], float(grid[h, a]),
                       f"Final NPZ grid cell ({h},{a}) vs prediction")


def _validate_scalar_params(results, cfg, metadata=None, methods=METHODS):
    """Validate the per-method final scalar-params exports and the combined
    comparison artifact. All four values must be finite; kappa and
    friendly_scale must be positive. The exports must use the final-epoch
    policy and the same dataset, and must agree with the fitted checkpoints.
    ``methods`` selects which per-method exports to validate (a partial run
    validates only its own method)."""
    per_method = {}
    for method in methods:
        scalars = _json(results / method / "final_scalar_params.json")
        _require(scalars.get("method") == method, f"{method}: scalar-params method mismatch")
        _require(scalars.get("checkpoint_policy") == "final_epoch", f"{method}: scalar-params wrong policy")
        _require(scalars.get("checkpoint_epoch") == cfg["n_epochs"], f"{method}: scalar-params wrong epoch")
        # Bind the scalar export to the actual checkpoint and dataset.
        fitted_path = results / method / "fitted_params.json"
        _nonempty(fitted_path)
        ckpt_hash = hashlib.sha256(fitted_path.read_bytes()).hexdigest()
        _require(scalars.get("checkpoint_hash") == ckpt_hash,
                 f"{method}: scalar-params checkpoint hash does not match the fitted checkpoint")
        _require(scalars.get("dataset_hash") == metadata.get("source_sha256"),
                 f"{method}: scalar-params dataset hash does not match the run metadata")
        params = scalars["parameters"]
        for name in ("alpha", "beta", "kappa", "friendly_scale"):
            _require(name in params, f"{method}: scalar-params missing {name}")
            _ensure_finite(f"{method} scalar {name}", float(params[name]))
        _require(float(params["kappa"]) > 0, f"{method}: kappa must be positive")
        _require(float(params["friendly_scale"]) > 0, f"{method}: friendly_scale must be positive")
        # Compare each scalar with the corresponding constrained checkpoint value.
        fitted = _json(fitted_path)
        if method == "ekf":
            ckpt = fitted["constrained"]
            ckpt_vals = {k: ckpt[k] for k in ("alpha", "beta", "kappa", "friendly_scale")}
        else:
            ckpt = fitted["constrained"]
            ckpt_vals = {"alpha": ckpt["model"]["alpha"], "beta": ckpt["model"]["beta"],
                         "kappa": ckpt["model"]["kappa"], "friendly_scale": ckpt["friendly_scale"]}
        for name in ("alpha", "beta", "kappa", "friendly_scale"):
            _close(params[name], float(ckpt_vals[name]), f"{method} scalar {name} vs checkpoint")
        per_method[method] = params
    # The combined comparison artifact is only present when both methods ran.
    if set(methods) == set(METHODS):
        combined = _json(results / "final_scalar_params_comparison.json")
        rows = combined["rows"]
        # Require the exact learned/allowed-derived row set, in order.
        expected_params = ["alpha", "beta", "kappa", "friendly_scale",
                           "exp(alpha)", "exp(beta)", "log(2)/kappa"]
        actual_params = [row["parameter"] for row in rows]
        _require(actual_params == expected_params,
                 "Scalar comparison must contain exactly the four learned parameters and the allowed derived rows")
        for row in rows:
            for field in ("ekf", "sqmc", "sqmc_minus_ekf"):
                _ensure_finite(f"scalar comparison {row['parameter']} {field}", float(row[field]))
            if row["parameter"] in ("alpha", "beta", "kappa", "friendly_scale"):
                _close(row["ekf"], per_method["ekf"][row["parameter"]], f"combined {row['parameter']} ekf vs export")
                _close(row["sqmc"], per_method["sqmc"][row["parameter"]], f"combined {row['parameter']} sqmc vs export")
                _close(row["sqmc_minus_ekf"], float(row["sqmc"]) - float(row["ekf"]),
                       f"combined {row['parameter']} sqmc_minus_ekf")
            elif row["parameter"] == "exp(alpha)":
                _close(row["ekf"], math.exp(per_method["ekf"]["alpha"]), "combined exp(alpha) ekf")
                _close(row["sqmc"], math.exp(per_method["sqmc"]["alpha"]), "combined exp(alpha) sqmc")
            elif row["parameter"] == "exp(beta)":
                _close(row["ekf"], math.exp(per_method["ekf"]["beta"]), "combined exp(beta) ekf")
                _close(row["sqmc"], math.exp(per_method["sqmc"]["beta"]), "combined exp(beta) sqmc")
            elif row["parameter"] == "log(2)/kappa":
                _close(row["ekf"], math.log(2) / per_method["ekf"]["kappa"], "combined log(2)/kappa ekf")
                _close(row["sqmc"], math.log(2) / per_method["sqmc"]["kappa"], "combined log(2)/kappa sqmc")
        for row in rows:
            _close(row["sqmc_minus_ekf"], float(row["sqmc"]) - float(row["ekf"]),
                   f"combined {row['parameter']} difference")
        # CSV must agree with the JSON rows.
        csv_rows = _csv(results / "final_scalar_params_comparison.csv")
        _require(len(csv_rows) == len(rows), "Scalar comparison CSV row count mismatch")
        for csv_row, json_row in zip(csv_rows, rows):
            _require(csv_row["parameter"] == json_row["parameter"], "Scalar comparison CSV parameter mismatch")
            for field in ("ekf", "sqmc", "sqmc_minus_ekf"):
                _close(float(csv_row[field]), float(json_row[field]), f"scalar comparison CSV {field}")
        _validate_scalar_table(results / "final_scalar_params_table.tex", rows)


def _validate_scalar_table(path, rows):
    """Validate complete ordered table rows and their model-column positions."""
    from rbsqmc.comparison.sqmc_ekf.scripts.scalar_table import table_body

    _nonempty(path)
    text = path.read_text()
    blocks = re.findall(r"\\begin\{tabular\}\{lccc\}(.*?)\\end\{tabular\}",
                        text, flags=re.DOTALL)
    _require(len(blocks) == 1 and text.count(r"\begin{tabular}") == 1,
             "Scalar table must contain exactly one four-column tabular")
    # Remove actual comments, not escaped percent signs in interpretations.
    body = re.sub(r"(?<!\\)%[^\n]*", "", blocks[0])
    actual = [" ".join(line.split()) for line in body.splitlines() if line.strip()]
    expected = [" ".join(line.split()) for line in table_body(rows)]
    _require(actual == expected,
             "Scalar table rows/Interpretation/model columns do not match the comparison")


def validate_artifacts(run_dir, config=None):
    """Validate exactly this run; optional config binds it to the launch request."""
    root = Path(run_dir)
    results, images = root / "results", root / "images"
    for filename in REQUIRED_RESULTS:
        _nonempty(results / filename)
    for path in results.rglob("*.json"):
        _json(path)
    cfg = _json(results / "comparison_config.json")
    if config is not None:
        _require(cfg == config, "Effective comparison configuration mismatch")
    epochs, goals = cfg["n_epochs"], cfg["max_goals"]
    _require(type(epochs) is int and epochs > 0, "Invalid configured epoch count")
    _require(type(goals) is int and goals >= 0, "Invalid configured score bound")
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for filename in ("run_config.json", "run_metadata.json"):
        provenance = _json(results / filename)
        _require(provenance["config_sha256"] == digest, f"{filename}: config digest mismatch")
        if "source_commit" in cfg:
            _require(provenance["source_commit"] == cfg["source_commit"], f"{filename}: source commit mismatch")
    metadata = _json(results / "dataset_metadata.json")
    run_metadata = _json(results / "run_metadata.json")
    count = metadata["prediction_count"]
    for field in ("train_count", "test_count", "prediction_count"):
        _require(type(metadata[field]) is int and metadata[field] > 0, f"Invalid {field}")
    _require(type(metadata["worldcup_count"]) is int and 0 <= metadata["worldcup_count"] <= count,
             "Invalid World Cup count")
    for field in ("prediction_count", "worldcup_count"):
        _require(run_metadata[field] == metadata[field], f"Run metadata {field} mismatch")
    summaries = _json(results / "summary.json")
    history = _csv(results / "logz_history.csv")
    performance = _csv(results / "performance_metrics.csv")
    _require(len(history) == 2 * epochs and {r["method"] for r in history} == set(METHODS),
             "History does not contain the configured epochs for both methods")
    _require(len(performance) == 2 and {r["method"] for r in performance} == set(METHODS),
             "Performance CSV must contain both methods exactly once")
    aligned_fixtures = None
    for method in METHODS:
        summary = _json(results / method / "summary.json")
        _require(summary == summaries[method], f"{method}: summary copies differ")
        _require(summary["n_epochs_completed"] == epochs, f"{method}: incomplete training")
        _require(summary["checkpoint_policy"] == "final_epoch", f"{method}: wrong checkpoint policy")
        _require(summary["learning_rate_schedule"] == "cosine", f"{method}: wrong optimizer schedule")
        _require(summary["gradient_replicas"] == (cfg["n_reps"] if method == "sqmc" else 1),
                 f"{method}: wrong replica count")
        rows = [r for r in history if r["method"] == method]
        _require([int(r["epoch"]) for r in rows] == list(range(1, epochs + 1)),
                 f"{method}: missing, duplicated or unordered epochs")
        for row in rows:
            _ensure_finite(f"{method} history", [float(row["train_logz"]), float(row["test_logz"])])
        best = max(rows, key=lambda row: float(row["test_logz"]))
        for field, value in (("final_train_logz", float(rows[-1]["train_logz"])),
                             ("final_test_logz", float(rows[-1]["test_logz"])),
                             ("best_test_logz", float(best["test_logz"])), ("best_test_epoch", int(best["epoch"]))):
            _close(summary[field], value, f"{method} {field}")
        records = _json(results / f"{method}_predictions.json")
        fixtures = _predictions(records, cfg, count)
        _require(aligned_fixtures is None or fixtures == aligned_fixtures, "Methods evaluated different fixtures or results")
        aligned_fixtures = fixtures
        wc = [r for r in records if r["worldcup_eligible"]]
        _require(len(wc) == metadata["worldcup_count"], f"{method}: World Cup count mismatch")
        metrics = _json(results / f"{method}_metrics.json")
        scored_wc = [r for r in wc if min(r["actual_home_score"], r["actual_away_score"]) >= 0]
        expected_metrics = dict(all=_metrics(records), worldcup=_metrics(scored_wc) if scored_wc else None)
        for group, expected in expected_metrics.items():
            if expected is None:
                _require(metrics[group] is None, f"{method}: metrics for unscored World Cup fixtures")
            else:
                for field, value in expected.items():
                    _close(metrics[group][field], value, f"{method} {group} {field}")
        row = next(r for r in performance if r["method"] == method)
        for field in row.keys() - {"method"}:
            if row[field] != "":
                _ensure_finite(f"{method} performance {field}", float(row[field]))
        headline = expected_metrics["worldcup"] or expected_metrics["all"]
        for field in ("mean_brier_score", "brier_skill_score_vs_uniform", "outcome_accuracy",
                      "exact_score_accuracy", "mean_log_likelihood", "n_scored"):
            _close(row[field], headline[field], f"{method} performance {field}")
        for field in ("final_train_logz", "final_test_logz", "best_test_logz"):
            _close(row[field], summary[field], f"{method} performance {field}")
        _close(row["compile_sec"], summary["compilation_sec"], f"{method} compilation time")
        _close(row["train_execution_sec"], summary["execution_sec"], f"{method} training time")
        times = [float(row[k]) for k in ("compile_sec", "train_execution_sec", "prediction_sec")]
        _require(all(t >= 0 for t in times), f"{method}: negative timing")
        _close(row["pipeline_sec"], sum(times), f"{method} pipeline time")
        for suffix in REQUIRED_METHOD_IMAGES:
            _nonempty(images / f"{method}_{suffix}")
    _validate_checkpoint(results, "ekf", cfg)
    _validate_checkpoint(results, "sqmc", cfg)
    sqmc_records = _json(results / "sqmc_predictions.json")
    _validate_sqmc_diagnostics(results, metadata, cfg, records=sqmc_records)
    _validate_scalar_params(results, cfg, metadata=metadata)
    _nonempty(images / "logz_overlay_train_test.png")
    for path in images.rglob("*.png"):
        _png(path)
    print(f"OK: comparison artifacts validated: {root}")


def _latest_run_dir(outputs_dir):
    candidates = []
    for path in Path(outputs_dir).iterdir():
        if path.is_dir():
            try:
                candidates.append((datetime.strptime(path.name, "%d%m%Y_%H%M"), path))
            except ValueError:
                continue
    _require(bool(candidates), f"No timestamped runs under {outputs_dir}")
    return max(candidates)[1]


def validate_run(path, config=None):
    root = Path(path)
    if config is None and not (root / "results").is_dir():
        root = _latest_run_dir(root)
    validate_artifacts(root, config)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "outputs"
    validate_run(path)


if __name__ == "__main__":
    main()

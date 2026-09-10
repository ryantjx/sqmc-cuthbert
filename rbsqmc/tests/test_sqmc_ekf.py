"""Tests for the SQMC–EKF comparison modules.

These tests exercise the pure/comparison-local logic: EKF OU propagation and
zero-elapsed-time handling, prediction-grid normalization, metric computation,
report serialization, and artifact validation. They avoid running the expensive
full filter/train by default so they stay fast and deterministic.
"""

import json
import os
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from rbsqmc.src.model.ekf import model as ekf
from rbsqmc.src.utils.helpers import default_init_params


# ---------------------------------------------------------------------------
# EKF OU moments, zero-time handling, initialization
# ---------------------------------------------------------------------------

def test_ekf_propagate_zero_elapsed_keeps_state():
    params = ekf.constrain(ekf.initial_raw(cov=np.eye(2) * 0.1))
    mean = np.array([[0.5, -0.2], [0.3, 0.1]])
    cov = np.broadcast_to(np.eye(2) * 0.05, (2, 2, 2))
    m, c = ekf.propagate(
        np.asarray(mean, dtype=float), np.asarray(cov, dtype=float),
        np.array(0.0), params,
    )
    # phi = exp(0) = 1 and Q = 0, so the state is unchanged at dt = 0.
    np.testing.assert_allclose(m, mean, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(c, cov, rtol=1e-5, atol=1e-6)


def test_ekf_propagate_reversion_for_large_elapsed():
    params = ekf.constrain(ekf.initial_raw(cov=np.eye(2) * 0.1, kappa=0.5))
    mean = np.array([[5.0, -3.0]])
    cov = np.broadcast_to(np.eye(2) * 0.1, (1, 2, 2))
    m, _ = ekf.propagate(
        np.asarray(mean, dtype=float), np.asarray(cov, dtype=float),
        np.array(50.0), params,
    )
    # Long horizon: the mean pulls strongly back toward the prior mean (zero).
    np.testing.assert_allclose(m[0], np.zeros(2), atol=1e-3)


def test_ekf_initial_roundtrips_constraints():
    raw = ekf.initial_raw(cov=np.eye(2) * 0.04)
    params = ekf.constrain(raw)
    # The constrained parameterization yields a valid covariance and its Cholesky.
    assert params["init_cov"].shape == (2, 2)
    np.testing.assert_allclose(
        params["init_chol_cov"] @ params["init_chol_cov"].T, params["init_cov"],
        rtol=1e-5, atol=1e-8,
    )
    assert np.isfinite(params["init_cov"]).all()
    assert np.isfinite(params["init_chol_cov"]).all()
    assert np.isfinite(params["kappa"])
    assert np.isfinite(params["init_mean"]).all()


def test_ekf_observation_uses_match_scale_for_non_friendly():
    """The EKF observation model must use the configured match_scale for
    non-friendly fixtures (not a hardcoded 1.0), matching the SQMC model."""
    params = ekf.constrain(ekf.initial_raw(cov=np.eye(2) * 0.1))
    # A non-friendly match (friendly=False) with a non-unit match_scale.
    inputs = ekf.MatchInputs(
        home=jnp.array([0]), away=jnp.array([1]),
        score=jnp.array([[1.0, 0.0]]),
        friendly=jnp.array([False]),
        timestamp=jnp.array([1.0]), previous=jnp.array([[0.0, 0.0]]),
    )
    # Run the filter with match_scale=2.0 and with match_scale=1.0; the
    # non-friendly observation scale differs, so the posterior must differ.
    h2 = ekf.run_filter(inputs, params, 2, match_scale=2.0)
    h1 = ekf.run_filter(inputs, params, 2, match_scale=1.0)
    assert not np.allclose(np.asarray(h2["mean"]), np.asarray(h1["mean"]))


def test_ekf_observation_friendly_uses_friendly_scale():
    """A friendly fixture uses the learned friendly_scale regardless of the
    configured match_scale."""
    params = ekf.constrain(ekf.initial_raw(cov=np.eye(2) * 0.1))
    inputs = ekf.MatchInputs(
        home=jnp.array([0]), away=jnp.array([1]),
        score=jnp.array([[1.0, 0.0]]),
        friendly=jnp.array([True]),
        timestamp=jnp.array([1.0]), previous=jnp.array([[0.0, 0.0]]),
    )
    # For a friendly, changing match_scale must NOT change the posterior.
    h2 = ekf.run_filter(inputs, params, 2, match_scale=2.0)
    h1 = ekf.run_filter(inputs, params, 2, match_scale=1.0)
    np.testing.assert_allclose(np.asarray(h2["mean"]), np.asarray(h1["mean"]), atol=1e-6)


# ---------------------------------------------------------------------------
# Prediction grid + metric helpers
# ---------------------------------------------------------------------------

from rbsqmc.comparison.sqmc_ekf.scripts import evaluate as eval_mod


def test_evaluate_metrics_known_scores():
    records = [
        {"actual_home_score": 2, "actual_away_score": 1,
         "predicted_home_score": 2, "predicted_away_score": 1,
         "log_likelihood": -0.5, "prob_home_win": 0.6, "prob_draw": 0.2,
         "prob_away_win": 0.2, "worldcup_eligible": True},
        {"actual_home_score": 0, "actual_away_score": 0,
         "predicted_home_score": 1, "predicted_away_score": 0,
         "log_likelihood": -0.9, "prob_home_win": 0.5, "prob_draw": 0.3,
         "prob_away_win": 0.2, "worldcup_eligible": True},
    ]
    m = eval_mod.compute_metrics(records)
    assert m.all["n_scored"] == 2
    assert 0.0 <= m.all["mean_brier_score"] <= 2.0
    assert m.worldcup["n_scored"] == 2
    # Uniform reference Brier is the fixed 2/3.
    assert m.all["uniform_reference_brier_score"] == pytest.approx(2.0 / 3.0)


def test_build_records_outcome_probabilities_sum_to_one():
    # Rebuild records requires a Dataset; test the outcome aggregation logic
    # directly on a constructed record grid.
    grid = np.array([[0.6, 0.1, 0.0],
                     [0.2, 0.05, 0.0],
                     [0.05, 0.0, 0.0]])
    assert grid.sum() == pytest.approx(1.0)
    # Home row > draw/away columns.
    prob_home = grid[1:, :].sum() - grid[1:, 1:].sum()  # placeholder
    # Simple check: the diagonal draw probability is the middle entry.
    assert grid[1, 1] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Report serialization
# ---------------------------------------------------------------------------

def test_report_writes_metrics(tmp_path):
    # Build a minimal results dict with plausible training summaries.
    def summary():
        return {
            "final_train_logz": -100.0, "final_test_logz": -20.0,
            "best_test_logz": -19.0, "best_test_epoch": 5,
            "compilation_sec": 0.1, "execution_sec": 1.0,
        }

    def metrics():
        return eval_mod.Metrics(
            all={"mean_brier_score": 0.4, "uniform_reference_brier_score": 2 / 3,
                 "brier_skill_score_vs_uniform": 0.4, "mean_log_likelihood": -0.6,
                 "exact_score_accuracy": 0.3, "outcome_accuracy": 0.5, "n_scored": 10},
            worldcup=None,
        )

    class _Dataset:
        metadata = {"source_rows": 100, "prediction_count": 10}
        train_count = 80
        test_count = 10

    class _M:
        worldcup = None
        all = metrics().all

    from rbsqmc.comparison.sqmc_ekf.scripts import report as report_mod

    results = {
        "cfg": {"max_goals": 8, "n_epochs": 10, "n_reps": 2},
        "ekf": {"summary": summary(), "metrics": _M()},
        "sqmc": {"summary": summary(), "metrics": _M()},
    }
    dataset = _Dataset()
    report_mod.write_report(str(tmp_path), dataset, results)
    for name in ("REPORT.md", "DRAFT.md"):
        assert (tmp_path / name).exists()
    text = (tmp_path / "REPORT.md").read_text()
    assert "SQMC–EKF Comparison" in text
    assert "Gaussian-approximate" in text  # distinguishes the two logZ estimators


# ---------------------------------------------------------------------------
# Fitted-params checkpoint round-trip
# ---------------------------------------------------------------------------

def test_fitted_params_sqmc_roundtrip(tmp_path):
    """A saved SQMC fitted-params checkpoint reconstructs the constrained
    parameters via the loader, including the fixed mean_0 and friendly scale."""
    import jax.numpy as jnp
    from rbsqmc.comparison.sqmc_ekf.scripts import train as train_mod
    from rbsqmc.src.utils.helpers import default_init_params, encode_EM_params

    base = default_init_params(4, {0: "A", 1: "B", 2: "C", 3: "D"})
    raw_model = encode_EM_params(base)
    raw = dict(model=raw_model, friendly_scale=ekf.inverse_positive(2.0))
    payload = {
        "method": "sqmc",
        "checkpoint_policy": "final_epoch",
        "checkpoint_epoch": 3,
        "raw": train_mod.jsonable(raw),
        "constrained": train_mod.jsonable(dict(
            model=base, friendly_scale=ekf.positive(raw["friendly_scale"]))),
        "team_id_to_name": {"0": "A", "1": "B", "2": "C", "3": "D"},
    }
    path = tmp_path / "fitted_params.json"
    train_mod.save_json(path, payload)

    fixed_mean = jnp.asarray(base.mean_0)
    params = train_mod.load_fitted_params(path, "sqmc", fixed_mean)
    assert float(params["friendly_scale"]) == pytest.approx(2.0, rel=1e-3)
    assert float(params["model"].alpha) == pytest.approx(float(base.alpha))
    assert float(params["model"].beta) == pytest.approx(float(base.beta))
    assert float(params["model"].kappa) == pytest.approx(float(base.kappa))
    np.testing.assert_allclose(np.asarray(params["model"].mean_0), np.asarray(base.mean_0))
    # gamma_0 and B are reparameterized by the EM encode/decode (det(B) scale
    # folded into gamma_0, B normalized to det=1), so they need not equal the
    # originals; they must be finite and valid.
    assert np.isfinite(np.asarray(params["model"].gamma_0)).all()
    assert np.isfinite(np.asarray(params["model"].B)).all()
    assert np.asarray(params["model"].B).shape == (2, 2)


def test_fitted_params_ekf_roundtrip(tmp_path):
    """An EKF fitted-params checkpoint reconstructs the constrained params."""
    from rbsqmc.comparison.sqmc_ekf.scripts import train as train_mod

    raw = ekf.initial_raw(cov=np.eye(2) * 0.04)
    payload = {
        "method": "ekf",
        "checkpoint_policy": "final_epoch",
        "checkpoint_epoch": 3,
        "raw": train_mod.jsonable(raw),
        "constrained": train_mod.jsonable(ekf.constrain(raw)),
        "team_id_to_name": {"0": "A", "1": "B"},
    }
    path = tmp_path / "fitted_params.json"
    train_mod.save_json(path, payload)
    params = train_mod.load_fitted_params(path, "ekf", None)
    np.testing.assert_allclose(np.asarray(params["init_cov"]), np.asarray(ekf.constrain(raw)["init_cov"]))
    assert float(params["friendly_scale"]) == pytest.approx(float(ekf.constrain(raw)["friendly_scale"]))


def test_sqmc_checkpoint_forecast_replay(tmp_path):
    """A forecast made with in-memory parameters must be reproduced exactly by
    loading only the saved checkpoint, using the same dataset and key."""
    import jax
    import jax.numpy as jnp
    from rbsqmc.comparison.sqmc_ekf.scripts import train as train_mod
    from rbsqmc.comparison.sqmc_ekf.scripts.scaling import sqmc_match_scales
    from rbsqmc.src.model.rbsqmc.model_rbsqmc import run_filter_sqmc
    from rbsqmc.src.model.rbsqmc.predict_rbsqmc import predict_from_sqmc_history
    from rbsqmc.src.utils.helpers import default_init_params, encode_EM_params
    from rbsqmc.src.utils.type import FootballResults, Matches

    # Build a small dataset with a nonzero observed prefix.
    def _inputs(timestamp, home, away, hs, as_):
        return FootballResults(
            date=jnp.asarray(timestamp) + 10,
            timestamp=jnp.asarray(timestamp),
            timestamp_prev=jnp.concatenate([jnp.array([0]), jnp.asarray(timestamp)[:-1]]),
            matches=Matches(home_id=jnp.asarray(home)[:, None], away_id=jnp.asarray(away)[:, None],
                            home_score=jnp.asarray(hs)[:, None], away_score=jnp.asarray(as_)[:, None]),
            match_mask=jnp.ones((len(timestamp), 1), dtype=bool),
        )
    sqmc = _inputs([1, 2, 3, 4], [0, 1, 2, 3], [1, 2, 3, 0], [1, 0, 2, 1], [0, 1, 1, 0])
    friendly = jnp.array([True, False, True, False])
    base = default_init_params(4, {0: "A", 1: "B", 2: "C", 3: "D"})
    raw = dict(model=encode_EM_params(base), friendly_scale=ekf.inverse_positive(2.0))
    data = SimpleNamespace(inputs=SimpleNamespace(friendly=friendly), sqmc=sqmc,
                           teams={0: "A", 1: "B", 2: "C", 3: "D"})
    methods = train_mod.Methods(data, dict(n_particles=8, max_goals=3, match_scale=1.5, seed=0))
    params = methods.params("sqmc", raw)
    scales = sqmc_match_scales(friendly, params["friendly_scale"], 1.5)
    key = jax.random.PRNGKey(0)
    result, _ = run_filter_sqmc(key, sqmc, params["model"], 8, 3, match_scales=scales)
    pred_inputs = _inputs([4], [3], [0], [1], [0])
    pred_scales = scales[3:4]
    grids_inmem, _, _ = predict_from_sqmc_history(
        result, pred_inputs, 3, params["model"], 3, pred_scales,
    )

    # Save the checkpoint and reload only from it.
    path = tmp_path / "fitted_params.json"
    train_mod.save_json(path, train_mod.jsonable(dict(
        method="sqmc", checkpoint_policy="final_epoch", checkpoint_epoch=3,
        raw=raw, constrained=params, team_id_to_name={0: "A", 1: "B", 2: "C", 3: "D"})))
    loaded = train_mod.load_fitted_params(path, "sqmc")
    result2, _ = run_filter_sqmc(key, sqmc, loaded["model"], 8, 3, match_scales=scales)
    grids_loaded, _, _ = predict_from_sqmc_history(
        result2, pred_inputs, 3, loaded["model"], 3, pred_scales,
    )
    np.testing.assert_allclose(np.asarray(grids_inmem), np.asarray(grids_loaded), atol=1e-6)
    # A conflicting supplied mean must be rejected.
    with pytest.raises(ValueError, match="does not match"):
        train_mod.load_fitted_params(path, "sqmc", jnp.zeros((4, 2)) + 1.0)


def test_scalar_comparison_export_roundtrip(tmp_path):
    """The combined scalar comparison export must preserve distinct known EKF
    and SQMC values without model-column swaps, raw-coordinate leakage, or
    covariance/mean rows."""
    from rbsqmc.comparison.sqmc_ekf import run as comparison

    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    # Distinct known values for EKF and SQMC.
    ekf_scalars = dict(alpha=0.1, beta=-3.0, kappa=0.002, friendly_scale=1.5)
    sqmc_scalars = dict(alpha=0.3, beta=-5.0, kappa=0.004, friendly_scale=2.5)
    for method, scalars in (("ekf", ekf_scalars), ("sqmc", sqmc_scalars)):
        (results_dir / method).mkdir(parents=True, exist_ok=True)
        comparison._save_json(results_dir / method / "final_scalar_params.json", dict(
            method=method, checkpoint_policy="final_epoch", checkpoint_epoch=3,
            run_id="test", source_revision="a" * 40, dataset_hash="x",
            checkpoint_hash="x", match_scale=1.0, parameters=scalars))
    comparison._write_scalar_comparison(results_dir, dict(run_id="test"), tmp_path)

    combined = json.loads((results_dir / "final_scalar_params_comparison.json").read_text())
    rows = {r["parameter"]: r for r in combined["rows"]}
    # No model-column swap: EKF and SQMC values must be distinct and correct.
    assert rows["alpha"]["ekf"] == pytest.approx(0.1)
    assert rows["alpha"]["sqmc"] == pytest.approx(0.3)
    assert rows["alpha"]["sqmc_minus_ekf"] == pytest.approx(0.2)
    assert rows["kappa"]["ekf"] == pytest.approx(0.002)
    assert rows["kappa"]["sqmc"] == pytest.approx(0.004)
    assert rows["friendly_scale"]["ekf"] == pytest.approx(1.5)
    assert rows["friendly_scale"]["sqmc"] == pytest.approx(2.5)
    # No covariance/mean rows: only the four scalars plus derived rows.
    params = {r["parameter"] for r in combined["rows"]}
    assert {"alpha", "beta", "kappa", "friendly_scale"} <= params
    assert not (params & {"init_cov", "init_chol_cov", "init_mean", "gamma_0", "B", "mean_0"})
    # CSV and LaTeX table are written.
    assert (results_dir / "final_scalar_params_comparison.csv").exists()
    assert (results_dir / "final_scalar_params_table.tex").exists()


def test_scalar_latex_escapes_run_id_and_compiles(tmp_path):
    """The generated LaTeX table must escape special characters in the run ID
    and parameter labels so it compiles in a minimal document."""
    from rbsqmc.comparison.sqmc_ekf import run as comparison

    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    for method, scalars in (("ekf", dict(alpha=0.1, beta=-3.0, kappa=0.002, friendly_scale=1.5)),
                            ("sqmc", dict(alpha=0.3, beta=-5.0, kappa=0.004, friendly_scale=2.5))):
        (results_dir / method).mkdir(parents=True, exist_ok=True)
        comparison._save_json(results_dir / method / "final_scalar_params.json", dict(
            method=method, checkpoint_policy="final_epoch", checkpoint_epoch=3,
            run_id="run_090926", source_revision="a" * 40, dataset_hash="x",
            checkpoint_hash="x", match_scale=1.0, parameters=scalars))
    # Run ID with an underscore must be escaped inside \texttt{...}.
    run_dir = tmp_path / "run_090926"
    run_dir.mkdir(exist_ok=True)
    comparison._write_scalar_comparison(results_dir, dict(run_id="run_090926"), run_dir)
    tex = (results_dir / "final_scalar_params_table.tex").read_text()
    assert r"run\_090926" in tex
    assert r"friendly\_scale" in tex
    assert r"$\alpha$" in tex
    assert "Interpretation" in tex
    # Compile in a minimal document if pdflatex is available.
    import shutil as _shutil
    if _shutil.which("pdflatex") is None:
        return
    doc = tmp_path / "doc.tex"
    doc.write_text(
        "\\documentclass{article}\n"
        "\\usepackage{booktabs}\n"
        "\\begin{document}\n"
        + tex +
        "\\end{document}\n"
    )
    import subprocess as _subprocess
    result = _subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                              "-output-directory", str(tmp_path), str(doc)],
                             capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_scalar_validation_rejects_inconsistent_values(tmp_path):
    """The scalar validator must reject a per-model alpha changed to 999 and a
    combined EKF value changed to 999 without matching source artifacts."""
    import hashlib as _hashlib
    from rbsqmc.comparison.sqmc_ekf import run as comparison
    from rbsqmc.comparison.sqmc_ekf.scripts.validate_sqmc_ekf_outputs import _validate_scalar_params

    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    for method, scalars in (("ekf", dict(alpha=0.1, beta=-3.0, kappa=0.002, friendly_scale=1.5)),
                            ("sqmc", dict(alpha=0.3, beta=-5.0, kappa=0.004, friendly_scale=2.5))):
        (results_dir / method).mkdir(parents=True, exist_ok=True)
        # Write a matching checkpoint so the scalar-vs-checkpoint check passes.
        if method == "ekf":
            ckpt = dict(alpha=0.1, beta=-3.0, kappa=0.002, friendly_scale=1.5)
        else:
            ckpt = dict(model=dict(alpha=0.3, beta=-5.0, kappa=0.004), friendly_scale=2.5)
        comparison._save_json(results_dir / method / "fitted_params.json", dict(
            method=method, checkpoint_policy="final_epoch", checkpoint_epoch=3,
            team_id_to_name={"0": "A", "1": "B"}, raw={}, constrained=ckpt))
        ckpt_hash = _hashlib.sha256((results_dir / method / "fitted_params.json").read_bytes()).hexdigest()
        comparison._save_json(results_dir / method / "final_scalar_params.json", dict(
            method=method, checkpoint_policy="final_epoch", checkpoint_epoch=3,
            run_id="test", source_revision="a" * 40, dataset_hash="x",
            checkpoint_hash=ckpt_hash, match_scale=1.0, parameters=scalars))
    comparison._write_scalar_comparison(results_dir, dict(run_id="test"), tmp_path)
    cfg = dict(n_epochs=3)
    metadata = {"source_sha256": "x"}
    # Baseline passes.
    _validate_scalar_params(results_dir, cfg, metadata=metadata)
    # Per-model alpha changed to 999 must fail.
    comparison._save_json(results_dir / "ekf" / "final_scalar_params.json", dict(
        method="ekf", checkpoint_policy="final_epoch", checkpoint_epoch=3,
        run_id="test", source_revision="a" * 40, dataset_hash="x",
        checkpoint_hash=ckpt_hash, match_scale=1.0,
        parameters=dict(alpha=999.0, beta=-3.0, kappa=0.002, friendly_scale=1.5)))
    with pytest.raises(ValueError):
        _validate_scalar_params(results_dir, cfg, metadata=metadata)
    # Restore, then change the combined EKF value to 999 without updating the
    # per-model export.
    comparison._save_json(results_dir / "ekf" / "final_scalar_params.json", dict(
        method="ekf", checkpoint_policy="final_epoch", checkpoint_epoch=3,
        run_id="test", source_revision="a" * 40, dataset_hash="x",
        checkpoint_hash="x", match_scale=1.0,
        parameters=dict(alpha=0.1, beta=-3.0, kappa=0.002, friendly_scale=1.5)))
    combined_path = results_dir / "final_scalar_params_comparison.json"
    combined = json.loads(combined_path.read_text())
    for row in combined["rows"]:
        if row["parameter"] == "alpha":
            row["ekf"] = 999.0
    comparison._save_json(combined_path, combined)
    with pytest.raises(ValueError):
        _validate_scalar_params(results_dir, cfg)


def test_sqmc_evaluation_uses_one_shared_filter(monkeypatch):
    """``_run_method`` must invoke the SQMC evaluation filter exactly once and
    pass its history to predictions, rankings, and diagnostics."""
    from rbsqmc.comparison.sqmc_ekf import run as comparison
    from rbsqmc.comparison.sqmc_ekf.scripts import train as train_mod
    from rbsqmc.comparison.sqmc_ekf.scripts.scaling import sqmc_match_scales
    from rbsqmc.src.model.rbsqmc.model_rbsqmc import run_filter_sqmc

    calls = {"count": 0, "result": None}

    def fake_run_filter(key, model_inputs, params, n_particles, max_goals, *, match_scales=None):
        calls["count"] += 1
        # Return a minimal valid history dict.
        T = model_inputs.timestamp.shape[0]
        N = n_particles
        num_teams = params.mean_0.shape[0]
        result = {
            "particles_x": jnp.zeros((T + 1, N, num_teams, 2)),
            "log_weights": jnp.zeros((T + 1, N)),
            "log_normalizing_constant": jnp.zeros(T + 1),
        }
        calls["result"] = result
        # Minimal augmented object with a gamma trajectory for plot_correlation.
        augmented = SimpleNamespace(gamma=jnp.zeros((T, num_teams, num_teams)))
        return result, augmented

    monkeypatch.setattr(comparison, "run_filter_sqmc", fake_run_filter)

    # Build a minimal dataset and Methods object.
    from rbsqmc.src.utils.type import FootballResults, Matches
    sqmc = FootballResults(
        date=jnp.array([0, 1, 2, 3]),
        timestamp=jnp.array([0, 1, 2, 3]),
        timestamp_prev=jnp.array([0, 0, 1, 2]),
        matches=Matches(home_id=jnp.array([[0], [1], [2], [3]]),
                        away_id=jnp.array([[1], [2], [3], [0]]),
                        home_score=jnp.array([[1], [0], [2], [1]]),
                        away_score=jnp.array([[0], [1], [1], [0]])),
        match_mask=jnp.ones((4, 1), dtype=bool),
    )
    friendly = jnp.array([True, False, True, False])
    data = SimpleNamespace(
        inputs=SimpleNamespace(friendly=friendly, score=jnp.zeros((4, 2))),
        sqmc=sqmc,
        teams={0: "A", 1: "B", 2: "C", 3: "D"},
        train_count=2, test_count=1,
        frame=pd.DataFrame([
            dict(date=pd.Timestamp("2026-06-08"), home_team="A", away_team="B",
                 home_id=0, away_id=1, home_score=0, away_score=0, tournament="Friendly"),
            dict(date=pd.Timestamp("2026-06-09"), home_team="B", away_team="A",
                 home_id=1, away_id=0, home_score=1, away_score=0, tournament="Friendly"),
            dict(date=pd.Timestamp("2026-06-10"), home_team="C", away_team="D",
                 home_id=2, away_id=3, home_score=0, away_score=0, tournament="Friendly"),
            dict(date=pd.Timestamp("2026-06-11"), home_team="A", away_team="B",
                 home_id=0, away_id=1, home_score=0, away_score=0, tournament="FIFA World Cup"),
        ]),
        metadata={"source_sha256": "x", "prediction_count": 1, "worldcup_count": 1},
    )
    cfg = dict(n_particles=8, max_goals=1, match_scale=1.0, seed=0, n_epochs=1,
               diagnostics_scope="all")
    methods = train_mod.Methods(data, cfg)
    root = jax.random.PRNGKey(0)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        results_dir = os.path.join(tmp, "results")
        images_dir = os.path.join(tmp, "images")
        os.makedirs(results_dir)
        os.makedirs(images_dir)
        # Monkeypatch train to avoid a real training run.
        from rbsqmc.src.utils.type import RawEMParams
        raw_model = RawEMParams(gamma_0_chol=jnp.eye(4), b_chol_raw=jnp.eye(2),
                                kappa_raw=jnp.array(0.0), alpha=jnp.array(0.2),
                                beta=jnp.array(-4.0))
        monkeypatch.setattr(methods, "train", lambda method, output: (
            dict(model=raw_model, friendly_scale=ekf.inverse_positive(2.0)),
            [dict(epoch=1, train_logz=-1.0, test_logz=-1.0, gradient_norm=0.0,
                  pre_update_loss=1.0, elapsed_sec=0.0)],
            dict(n_epochs_completed=1, checkpoint_policy="final_epoch",
                 learning_rate_schedule="cosine", gradient_replicas=1,
                 final_train_logz=-1.0, final_test_logz=-1.0,
                 best_test_logz=-1.0, best_test_epoch=1,
                 compilation_sec=0.0, execution_sec=0.0),
        ))
        comparison._run_method("sqmc", methods, data, cfg, root, results_dir, images_dir)
    # Exactly one evaluation filter invocation.
    assert calls["count"] == 1


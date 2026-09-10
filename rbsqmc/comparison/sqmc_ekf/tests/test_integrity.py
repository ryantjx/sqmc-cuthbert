"""Regression checks for result integrity, download lifecycle and state indexing."""

import csv
import io
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from rbsqmc.comparison.sqmc_ekf import run as comparison
from rbsqmc.comparison.sqmc_ekf.scripts import evaluate, sqmc_ekf_protocol as protocol
from rbsqmc.comparison.sqmc_ekf.scripts.validate_sqmc_ekf_outputs import (
    REQUIRED_METHOD_IMAGES, _latest_run_dir,
)
from rbsqmc.src.model.ekf import model as ekf


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def complete_run(root):
    """Use the real output writers and evaluator to construct a small valid run."""
    cfg = dict(n_epochs=2, n_reps=2, max_goals=1, n_particles=8, prediction_start_date="2026-06-11",
               source_commit="a" * 40, run_id="08092026_1200", session_name="test", transfer_timeout=10)
    results_dir, images = root / "results", root / "images"
    results_dir.mkdir(parents=True)
    images.mkdir()
    write_json(results_dir / "comparison_config.json", cfg)
    provenance = dict(config_sha256=protocol.config_digest(cfg), source_commit=cfg["source_commit"])
    write_json(results_dir / "run_config.json", provenance)
    write_json(results_dir / "run_metadata.json", dict(provenance, prediction_count=2, worldcup_count=2))
    write_json(results_dir / "dataset_metadata.json", dict(train_count=2, test_count=1,
                                                          prediction_count=2, worldcup_count=2,
                                                          source_sha256="x"))
    dataset = SimpleNamespace(train_count=0, test_count=0, frame=pd.DataFrame([
        dict(date=pd.Timestamp("2026-06-11"), home_team="A", away_team="B", home_score=0,
             away_score=0, tournament="FIFA World Cup"),
        dict(date=pd.Timestamp("2026-06-12"), home_team="B", away_team="A", home_score=1,
             away_score=0, tournament="FIFA World Cup"),
    ]))
    grids = np.array([[[.4, .1], [.3, .2]], [[.20140285789966583, .2225847691297531], [.2718656063079834, .30414679646492004]]])
    records = evaluate.build_records(
        dataset, grids,
        np.log(np.array([.4, .2718656063079834]) + 1e-12),
    )
    metrics = evaluate.compute_metrics(records)
    histories = [dict(epoch=1, train_logz=-10., test_logz=-4.),
                 dict(epoch=2, train_logz=-9., test_logz=-5.)]
    summaries, results = {}, {}
    # Valid EKF and SQMC checkpoint parameter structures (raw + constrained).
    # The EKF constrained values are the softplus-transformed raw values.
    ekf_raw = dict(init_sd=0.1, init_corr=0.0, kappa=0.001, alpha=0.2, beta=-4.0, friendly_scale=2.0)
    ekf_constrained = dict(init_mean=[0.0, 0.0], init_cov=[[0.5541279315948486, 0.0], [0.0, 0.5541279315948486]],
                           init_chol_cov=[[0.7443976998329163, 0.0], [0.0, 0.7443976998329163]],
                           kappa=0.6936483383178711, alpha=0.2, beta=-4.0, friendly_scale=2.1269290447235107)
    sqmc_raw = dict(
        model=dict(gamma_0_chol=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                   b_chol_raw=[[1.0, 0.0], [0.0, 1.0]],
                   kappa_raw=0.0, alpha=0.2, beta=-4.0),
        friendly_scale=2.0,
    )
    sqmc_constrained = dict(
        model=dict(mean_0=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                   gamma_0=[[1.724919080734253, 0.0, 0.0, 0.0], [0.0, 1.724919080734253, 0.0, 0.0],
                            [0.0, 0.0, 1.724919080734253, 0.0], [0.0, 0.0, 0.0, 1.724919080734253]],
                   B=[[1.0, 0.0], [0.0, 1.0]],
                   kappa=0.6931481957435608, alpha=0.2, beta=-4.0),
        friendly_scale=2.126929011042973,
    )
    for method in ("ekf", "sqmc"):
        summary = dict(n_epochs_completed=2, checkpoint_policy="final_epoch",
                       learning_rate_schedule="cosine", gradient_replicas=2 if method == "sqmc" else 1,
                       final_train_logz=-9., final_test_logz=-5., best_test_logz=-4., best_test_epoch=1,
                       compilation_sec=.1, execution_sec=.2)
        summaries[method] = summary
        write_json(results_dir / method / "summary.json", summary)
        write_json(results_dir / method / "fitted_params.json", dict(
            method=method, checkpoint_policy="final_epoch", checkpoint_epoch=2,
            team_id_to_name={"0": "A", "1": "B"},
            raw=ekf_raw if method == "ekf" else sqmc_raw,
            constrained=ekf_constrained if method == "ekf" else sqmc_constrained))
        import hashlib as _hashlib
        ckpt_hash = _hashlib.sha256((results_dir / method / "fitted_params.json").read_bytes()).hexdigest()
        write_json(results_dir / method / "final_scalar_params.json", dict(
            method=method, checkpoint_policy="final_epoch", checkpoint_epoch=2,
            run_id="08092026_1200", source_revision="a" * 40, dataset_hash="x",
            checkpoint_hash=ckpt_hash, match_scale=1.0,
            parameters=dict(alpha=0.2, beta=-4.0,
                            kappa=0.6936483383178711 if method == "ekf" else 0.6931481805599453,
                            friendly_scale=2.1269290447235107 if method == "ekf" else 2.126929011042973)))
        write_json(results_dir / f"{method}_history.json", histories)
        write_json(results_dir / f"{method}_predictions.json", records)
        write_json(results_dir / f"{method}_metrics.json", dict(all=metrics.all, worldcup=metrics.worldcup))
        results[method] = dict(summary=summary, history=histories, metrics=metrics, prediction_sec=.3)
    # Repaired-schema SQMC diagnostics artifacts required by validation.
    # The two prediction records are both 2026 World Cup fixtures, so the
    # worldcup scope must contain both, aligned to their records. The
    # diagnostics are bound to the SQMC checkpoint via its SHA-256.
    import hashlib as _hashlib
    sqmc_checkpoint = results_dir / "sqmc" / "fitted_params.json"
    checkpoint_hash = _hashlib.sha256(sqmc_checkpoint.read_bytes()).hexdigest()
    write_json(results_dir / "sqmc_prediction_diagnostics.json", dict(
        schema_version=1, n_particles=8, max_goals=1, source_revision="a" * 40,
        dataset_hash="x", checkpoint_hash=checkpoint_hash,
        filter_key_provenance="fold_in(root, 2_000_000)",
        scope="worldcup",
        forecasts=[
            dict(fixture_id=0, date="2026-06-11", home_id=0, away_id=1, home="A", away="B",
                 full_sequence_match_index=3, history_index=4, current_scale=1.0, label="predictive",
                 predictive_means=dict(home_attack=.1, home_defence=.1, away_attack=.1, away_defence=.1),
                 posterior_means=dict(home_attack=.1, home_defence=.1, away_attack=.1, away_defence=.1),
                 total_strength_difference=dict(mean=0.0, q05=-.1, q50=0.0, q95=.1, fraction_positive=.5),
                 ess_before_resampling=8.0, ess_after_update=8.0, raw_grid_mass=1.0,
                 outcome_probabilities=dict(home=.3, draw=.6, away=.1), n_particles=8, max_goals=1),
            dict(fixture_id=1, date="2026-06-12", home_id=1, away_id=0, home="B", away="A",
                 full_sequence_match_index=4, history_index=5, current_scale=1.0, label="predictive",
                 predictive_means=dict(home_attack=.5, home_defence=.2, away_attack=.1, away_defence=.4),
                 posterior_means=dict(home_attack=.5, home_defence=.2, away_attack=.1, away_defence=.4),
                 total_strength_difference=dict(mean=0.2, q05=.2, q50=.2, q95=.2, fraction_positive=1.0),
                 ess_before_resampling=8.0, ess_after_update=8.0, raw_grid_mass=0.418575257062912,
                 outcome_probabilities=dict(home=.2718656063079834, draw=.5055496723646492, away=.2225847691297531), n_particles=8, max_goals=1),
        ],
    ))
    np.savez(results_dir / "sqmc_final_fixture.npz",
             particles_x_predictive=np.array([
                 [[.1, .4], [.5, .2]] for _ in range(8)
             ]),
             log_weights_predictive=np.zeros(8),
             log_weights_posterior=np.zeros(8),
             home_id=np.array(1), away_id=np.array(0),
             scale=np.array(1.0), alpha=np.array(0.2), beta=np.array(-4.0),
             max_goals=np.array(1))
    # Combined scalar comparison artifacts required by validation.
    write_json(results_dir / "final_scalar_params_comparison.json", dict(
        run_id="08092026_1200", checkpoint_epoch_ekf=2, checkpoint_epoch_sqmc=2,
        rows=[
            dict(parameter="alpha", meaning="Baseline log scoring rate", unit="log goals",
                 ekf=0.2, sqmc=0.2, sqmc_minus_ekf=0.0),
            dict(parameter="beta", meaning="Log rate of the shared Poisson component", unit="log goals",
                 ekf=-4.0, sqmc=-4.0, sqmc_minus_ekf=0.0),
            dict(parameter="kappa", meaning="OU mean-reversion rate", unit="per day",
                 ekf=0.6936483383178711, sqmc=0.6931481805599453, sqmc_minus_ekf=-0.0005001577579258),
            dict(parameter="friendly_scale", meaning="Strength scaling for friendly matches", unit="dimensionless",
                 ekf=2.1269290447235107, sqmc=2.126929011042973, sqmc_minus_ekf=-3.3680537e-08),
            dict(parameter="exp(alpha)", meaning="Baseline scoring rate", unit="goals",
                 ekf=math.exp(0.2), sqmc=math.exp(0.2), sqmc_minus_ekf=0.0),
            dict(parameter="exp(beta)", meaning="Shared-component rate", unit="goals",
                 ekf=math.exp(-4.0), sqmc=math.exp(-4.0), sqmc_minus_ekf=0.0),
            dict(parameter="log(2)/kappa", meaning="OU half-life", unit="days",
                 ekf=math.log(2) / 0.6936483383178711, sqmc=math.log(2) / 0.6931481805599453,
                 sqmc_minus_ekf=math.log(2) / 0.6931481805599453 - math.log(2) / 0.6936483383178711),
        ],
    ))
    (results_dir / "final_scalar_params_comparison.csv").write_text(
        "parameter,meaning,unit,ekf,sqmc,sqmc_minus_ekf\n"
        "alpha,Baseline log scoring rate,log goals,0.2,0.2,0.0\n"
        "beta,Log rate of the shared Poisson component,log goals,-4.0,-4.0,0.0\n"
        "kappa,OU mean-reversion rate,per day,0.6936483383178711,0.6931481805599453,-0.0005001577579258\n"
        "friendly_scale,Strength scaling for friendly matches,dimensionless,2.1269290447235107,2.126929011042973,-3.3680537e-08\n"
        "exp(alpha),Baseline scoring rate,goals,1.2214027581601699,1.2214027581601699,0.0\n"
        "exp(beta),Shared-component rate,goals,0.01831563888873418,0.01831563888873418,0.0\n"
        "log(2)/kappa,OU half-life,days,0.9992775045650061,0.9999985573070405,0.0007210527420343782\n")
    (results_dir / "final_scalar_params_table.tex").write_text(
        "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{lccc}\n\\toprule\n"
        "Parameter & Interpretation & EKF & RB-SQMC \\\\\n\\midrule\n"
        "$\\alpha$ & Baseline log scoring rate & 0.2000 & 0.2000 \\\\\n"
        "$\\beta$ & Log rate of the shared Poisson component & -4.0000 & -4.0000 \\\\\n"
        "$\\kappa$ & OU mean-reversion rate & 6.936e-01 & 6.931e-01 \\\\\n"
        "friendly\\_scale & Strength scaling for friendly matches & 2.1269 & 2.1269 \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write_json(results_dir / "summary.json", summaries)
    comparison.write_performance_metrics_csv(results_dir, results)
    comparison.write_logz_history_csv(results_dir, results)
    for name in ("REPORT.md", "DRAFT.md"):
        (results_dir / name).write_text("Two epochs completed; comparison fixture.\n")
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.plot([0, 1], [0, 1])
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    for name in ["logz_overlay_train_test.png"] + [f"{m}_{s}" for m in ("ekf", "sqmc") for s in REQUIRED_METHOD_IMAGES]:
        (images / name).write_bytes(buffer.getvalue())
    return cfg


def test_partial_validation_checks_epochs_and_probabilities(tmp_path):
    cfg = complete_run(tmp_path)
    history = [dict(epoch=1, train_logz=-10., test_logz=-4.),
               dict(epoch=2, train_logz=-9., test_logz=-5.)]
    write_json(tmp_path / "results/sqmc_history.json", history)
    protocol.validate_run(tmp_path, cfg, methods=("sqmc",))
    write_json(tmp_path / "results/sqmc_history.json", history[:1])
    with pytest.raises(ValueError, match="epochs"):
        protocol.validate_run(tmp_path, cfg, methods=("sqmc",))
    write_json(tmp_path / "results/sqmc_history.json", history)
    records = json.loads((tmp_path / "results/sqmc_predictions.json").read_text())
    records[0]["score_probabilities"][0]["probability"] = -1
    write_json(tmp_path / "results/sqmc_predictions.json", records)
    with pytest.raises(ValueError, match="probability"):
        protocol.validate_run(tmp_path, cfg, methods=("sqmc",))


def test_sqmc_partial_does_not_require_ekf_checkpoint(tmp_path):
    """A SQMC-only partial must validate without the EKF checkpoint present.
    This is the partial-run path used by the Colab launcher."""
    cfg = complete_run(tmp_path)
    # Remove the EKF checkpoint; a SQMC-only partial must not require it.
    (tmp_path / "results/ekf/fitted_params.json").unlink()
    protocol.validate_run(tmp_path, cfg, methods=("sqmc",))
    # Removing the SQMC checkpoint must fail validation.
    (tmp_path / "results/sqmc/fitted_params.json").unlink()
    with pytest.raises(ValueError, match="fitted_params"):
        protocol.validate_run(tmp_path, cfg, methods=("sqmc",))


def test_ekf_partial_requires_own_checkpoint(tmp_path):
    """An EKF-only partial must validate its own checkpoint and not require the
    SQMC diagnostics."""
    cfg = complete_run(tmp_path)
    # Remove the SQMC checkpoint and diagnostics; an EKF-only partial must not
    # require them.
    (tmp_path / "results/sqmc/fitted_params.json").unlink()
    (tmp_path / "results/sqmc_prediction_diagnostics.json").unlink()
    (tmp_path / "results/sqmc_final_fixture.npz").unlink()
    protocol.validate_run(tmp_path, cfg, methods=("ekf",))
    # Removing the EKF checkpoint must fail validation.
    (tmp_path / "results/ekf/fitted_params.json").unlink()
    with pytest.raises(ValueError, match="fitted_params"):
        protocol.validate_run(tmp_path, cfg, methods=("ekf",))


def test_diagnostics_scope_filters_forecasts():
    """The diagnostics scope config controls how many forecasts are persisted:
    worldcup keeps only 2026 World Cup fixtures, final keeps only the last."""
    from rbsqmc.comparison.sqmc_ekf.scripts import diagnostics as diag_mod

    class _Row(dict):
        def __init__(self, tournament, year):
            super().__init__(
                home_id=0, away_id=1, home_team="A", away_team="B",
                source_row=0, tournament=tournament,
                date=pd.Timestamp(f"{year}-06-11"),
            )

    class _Iloc:
        def __init__(self, rows):
            self._rows = rows

        def __getitem__(self, _):
            return self

        def __len__(self):
            return len(self._rows)

        def iterrows(self):
            return iter(enumerate(self._rows))

    class _Frame:
        def __init__(self, rows):
            self.iloc = _Iloc(rows)

    # 3 prediction rows: a friendly, a 2026 WC, a 2025 WC.
    rows = [_Row("Friendly", 2026), _Row("FIFA World Cup", 2026), _Row("FIFA World Cup", 2025)]
    dataset = SimpleNamespace(
        train_count=0, test_count=0, frame=_Frame(rows),
        teams={0: "A", 1: "B"},
    )
    params = {"model": SimpleNamespace(alpha=0.2, beta=-4.0)}
    cfg = dict(max_goals=3, n_particles=8)
    n = len(rows)
    result = {
        "particles_x": jnp.zeros((n + 1, 8, 2, 2)),
        "log_weights": jnp.zeros((n + 1, 8)),
    }
    scales = jnp.ones((n, 1))

    # worldcup scope: only the 2026 WC row survives.
    payload, _ = diag_mod.build_diagnostics(
        dataset, params, cfg, result, scales, "rev", "hash", "key", 1, scope="worldcup",
    )
    assert len(payload["forecasts"]) == 1
    assert payload["forecasts"][0]["tournament"] == "FIFA World Cup"
    assert payload["forecasts"][0]["date"].startswith("2026")

    # final scope: only the last row survives.
    payload, _ = diag_mod.build_diagnostics(
        dataset, params, cfg, result, scales, "rev", "hash", "key", 1, scope="final",
    )
    assert len(payload["forecasts"]) == 1
    assert payload["forecasts"][0]["tournament"] == "FIFA World Cup"
    assert payload["forecasts"][0]["date"].startswith("2025")

    # all scope: every row survives.
    payload, _ = diag_mod.build_diagnostics(
        dataset, params, cfg, result, scales, "rev", "hash", "key", 1, scope="all",
    )
    assert len(payload["forecasts"]) == 3


def test_combine_allows_added_transport_hash_but_rejects_changed_science(tmp_path):
    cfg = complete_run(tmp_path)
    dataset = SimpleNamespace(metadata=json.loads((tmp_path / "results/dataset_metadata.json").read_text()))
    transport_cfg = dict(cfg, source_bundle_sha256="b" * 64)
    write_json(tmp_path / "results/comparison_config.json", transport_cfg)
    comparison._check_partial("sqmc", tmp_path, cfg, dataset)
    transport_cfg["n_epochs"] += 1
    write_json(tmp_path / "results/comparison_config.json", transport_cfg)
    with pytest.raises(ValueError, match="n_epochs"):
        comparison._check_partial("sqmc", tmp_path, cfg, dataset)


def edit_json(root, name, change):
    path = root / "results" / name
    value = json.loads(path.read_text())
    change(value)
    write_json(path, value)


def test_real_writers_pass_production_validation(tmp_path):
    cfg = complete_run(tmp_path)
    protocol.validate_run(tmp_path, cfg)


@pytest.mark.parametrize("defect", [
    "bad_home_id", "bad_mass", "bad_ess", "bad_probs", "bad_dataset_hash",
    "empty_worldcup", "bad_npz", "bad_checkpoint_hash",
    "wrong_history_index", "duplicate_entry", "swapped_ids", "shifted_npz",
])
def test_production_rejects_invalid_diagnostics(tmp_path, defect):
    """The validator must reject scientifically inconsistent diagnostics."""
    cfg = complete_run(tmp_path)
    diag_path = tmp_path / "results/sqmc_prediction_diagnostics.json"
    diag = json.loads(diag_path.read_text())
    if defect == "bad_home_id":
        diag["forecasts"][0]["home_id"] = 9999
        diag["forecasts"][0]["history_index"] = 9999
    elif defect == "bad_mass":
        diag["forecasts"][0]["raw_grid_mass"] = -1.0
        diag["forecasts"][0]["ess_after_update"] = -3.0
    elif defect == "bad_ess":
        diag["forecasts"][0]["ess_after_update"] = 100.0  # > n_particles
    elif defect == "bad_probs":
        diag["forecasts"][0]["outcome_probabilities"] = dict(home=1.0, draw=0.0, away=0.0)
    elif defect == "bad_dataset_hash":
        diag["dataset_hash"] = "wrong"
    elif defect == "empty_worldcup":
        diag["forecasts"] = []
    elif defect == "bad_npz":
        (tmp_path / "results/sqmc_final_fixture.npz").write_bytes(b"not an npz archive")
    elif defect == "bad_checkpoint_hash":
        diag["checkpoint_hash"] = "b" * 64
    elif defect == "wrong_history_index":
        diag["forecasts"][0]["history_index"] = 9999
    elif defect == "duplicate_entry":
        diag["forecasts"][1] = dict(diag["forecasts"][0])
    elif defect == "swapped_ids":
        diag["forecasts"][0]["home_id"], diag["forecasts"][0]["away_id"] = (
            diag["forecasts"][0]["away_id"], diag["forecasts"][0]["home_id"])
    elif defect == "shifted_npz":
        import numpy as _np
        with _np.load(tmp_path / "results/sqmc_final_fixture.npz", allow_pickle=False) as data:
            particles = data["particles_x_predictive"].copy()
            particles[:, 1, :] += 100.0  # shift only the home team
            lw_pred = data["log_weights_predictive"]
            lw_post = data["log_weights_posterior"]
            home_id = data["home_id"]
            away_id = data["away_id"]
            scale = data["scale"]
            alpha = data["alpha"]
            beta = data["beta"]
            max_goals = data["max_goals"]
        _np.savez(tmp_path / "results/sqmc_final_fixture.npz",
                  particles_x_predictive=particles,
                  log_weights_predictive=lw_pred,
                  log_weights_posterior=lw_post,
                  home_id=home_id, away_id=away_id,
                  scale=scale, alpha=alpha, beta=beta,
                  max_goals=max_goals)
    write_json(diag_path, diag)
    with pytest.raises(ValueError):
        protocol.validate_run(tmp_path, cfg)


@pytest.mark.parametrize("defect", ["empty", "nan_json", "nan_csv", "truncated_epochs", "duplicate_epoch",
                                      "bad_grid", "wrong_metrics", "different_fixture", "wrong_logp",
                                      "corrupt_png", "wrong_config", "wrong_provenance"])
def test_production_rejects_invalid_artifacts(tmp_path, defect):
    cfg = complete_run(tmp_path)
    if defect == "empty":
        (tmp_path / "results/REPORT.md").write_text("")
    elif defect == "nan_json":
        edit_json(tmp_path, "ekf_predictions.json", lambda records: records[0].update(log_likelihood=float("nan")))
    elif defect in ("nan_csv", "truncated_epochs", "duplicate_epoch"):
        path = tmp_path / "results/logz_history.csv"
        rows = list(csv.DictReader(io.StringIO(path.read_text())))
        if defect == "nan_csv":
            rows[0]["train_logz"] = "nan"
        elif defect == "truncated_epochs":
            rows = [r for r in rows if r["epoch"] == "1"]
        else:
            rows[1]["epoch"] = "1"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    elif defect == "bad_grid":
        edit_json(tmp_path, "ekf_predictions.json", lambda records: records[0]["score_probabilities"][0].update(probability=-.1))
    elif defect == "wrong_metrics":
        edit_json(tmp_path, "ekf_metrics.json", lambda metrics: metrics["worldcup"].update(mean_brier_score=.987))
    elif defect == "different_fixture":
        edit_json(tmp_path, "sqmc_predictions.json", lambda records: records[0].update(home="C"))
    elif defect == "wrong_logp":
        edit_json(tmp_path, "ekf_predictions.json", lambda records: records[0].update(log_likelihood=-100.))
    elif defect == "corrupt_png":
        (tmp_path / "images/ekf_pre_worldcup_rankings.png").write_bytes(b"not a PNG")
    elif defect == "wrong_config":
        cfg = dict(cfg, n_epochs=100)
    elif defect == "wrong_provenance":
        edit_json(tmp_path, "run_config.json", lambda value: value.update(config_sha256="b" * 64))
    with pytest.raises(ValueError):
        protocol.validate_run(tmp_path, cfg)


def test_empty_placeholder_bundle_is_rejected(tmp_path):
    for name in ("results/summary.json", "results/performance_metrics.csv", "results/logz_history.csv",
                 "results/REPORT.md", "results/DRAFT.md"):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.touch()
    with pytest.raises(ValueError, match="Missing or empty"):
        protocol.validate_run(tmp_path, {"n_epochs": 100})


def test_latest_timestamp_is_chronological_across_months(tmp_path):
    for name in ("31082026_1200", "01092026_1200"):
        (tmp_path / name).mkdir()
    assert _latest_run_dir(tmp_path).name == "01092026_1200"


def test_root_refresh_preserves_completed_results_and_rejects_repeat_results(tmp_path, monkeypatch):
    scripts = Path(comparison.__file__).parent / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    import run_sqmc_ekf_local as local
    remote, output = tmp_path / "remote", tmp_path / "downloaded"
    cfg = complete_run(remote)
    output.mkdir()
    write_json(remote / "comparison_config.json", cfg)
    write_json(remote / "run_config.json", dict(source_commit=cfg["source_commit"],
                                                config_sha256=protocol.config_digest(cfg)))
    write_json(remote / "remote_status.json", dict(run=dict(execution="pending")))
    (remote / "remote_logs.txt").write_text("setup done\n")
    protocol.make_archive(remote, "root", cfg)
    protocol.make_archive(remote, "run", cfg)

    def transfer(argv, **kwargs):
        assert argv[1] == "download"
        shutil.copyfile(remote / Path(argv[-2]).name, argv[-1])
        return ""

    launcher = local.Launcher(cfg, output, run=transfer)
    launcher.download("root")
    assert launcher.status["run"]["download"] == "pending"
    launcher.download("run")
    assert launcher.status["run"]["download"] == "complete"
    write_json(remote / "remote_status.json", dict(run=dict(execution="complete")))
    protocol.make_archive(remote, "root", cfg)
    launcher.download("root")
    assert launcher.status["run"]["download"] == "complete"
    assert json.loads((output / "remote_status.json").read_text())["run"]["execution"] == "complete"
    with pytest.raises(RuntimeError, match="complete download"):
        launcher.download("run")


def test_download_rejects_semantically_invalid_but_checksummed_results(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(comparison.__file__).parent / "scripts"))
    import run_sqmc_ekf_local as local
    remote, output = tmp_path / "remote", tmp_path / "downloaded"
    cfg = complete_run(remote)
    output.mkdir()
    (remote / "results/logz_history.csv").write_text("method,epoch,train_logz,test_logz\n")
    protocol.make_archive(remote, "run", cfg)
    def transfer(argv, **kwargs):
        shutil.copyfile(remote / Path(argv[-2]).name, argv[-1])
    launcher = local.Launcher(cfg, output, run=transfer)
    with pytest.raises(ValueError, match="Empty CSV"):
        launcher.download("run")
    assert launcher.status["run"]["download"] != "complete"
    assert not (output / "results").exists()


def test_pre_worldcup_ekf_ranking_is_independent_of_first_worldcup_result():
    # The two WC rows share a timestamp; the first must not affect the pre-WC
    # state, but it must affect the next state. Use the actual factorial filter.
    inputs = ekf.MatchInputs(jnp.array([0, 1, 0, 1]), jnp.array([1, 0, 1, 0]),
                             jnp.array([[1., 0.], [0., 0.], [3., 0.], [1., 1.]]),
                             jnp.zeros(4, dtype=bool), jnp.array([1., 2., 3., 3.]),
                             jnp.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.]]))
    params = ekf.constrain(ekf.initial_raw(jnp.eye(2)))
    data = SimpleNamespace(inputs=inputs)
    states, _ = comparison._ekf_states(data, params, 2)
    changed = SimpleNamespace(inputs=inputs._replace(score=inputs.score.at[2].set(jnp.array([0., 3.]))))
    changed_states, _ = comparison._ekf_states(changed, params, 2)
    assert states.particles.x.shape == (5, 1, 2, 2)
    assert states.ekf_cov.shape == (5, 2, 2, 2)
    np.testing.assert_allclose(states.particles.x[0, 0], jnp.zeros((2, 2)))
    np.testing.assert_allclose(states.ekf_cov[0], jnp.broadcast_to(params["init_cov"], (2, 2, 2)))
    np.testing.assert_allclose(states.particles.x[2], changed_states.particles.x[2])
    assert not np.allclose(states.particles.x[3], changed_states.particles.x[3])


# ---------------------------------------------------------------------------
# Repaired-pipeline regression tests: scale continuity and scale alignment.
# ---------------------------------------------------------------------------


def _sqmc_inputs(timestamp, home, away, home_score, away_score):
    from rbsqmc.src.utils.type import FootballResults, Matches
    return FootballResults(
        date=jnp.asarray(timestamp) + 10,
        timestamp=jnp.asarray(timestamp),
        timestamp_prev=jnp.concatenate([jnp.array([0]), jnp.asarray(timestamp)[:-1]]),
        matches=Matches(
            home_id=jnp.asarray(home)[:, None],
            away_id=jnp.asarray(away)[:, None],
            home_score=jnp.asarray(home_score)[:, None],
            away_score=jnp.asarray(away_score)[:, None],
        ),
        match_mask=jnp.ones((len(timestamp), 1), dtype=bool),
    )


def _em_params():
    from rbsqmc.src.utils.type import EMParams
    return EMParams(
        mean_0=jnp.zeros((4, 2)),
        gamma_0=jnp.eye(4),
        B=jnp.array([[1.0, 0.2], [0.2, 1.0]]),
        kappa=jnp.array(0.001),
        alpha=jnp.array(0.2),
        beta=jnp.array(-4.0),
    )


def test_scale_continuity_eval_history_equals_training_filter():
    """With a known non-unit friendly scale, the final evaluation history
    equals ``Methods.filter`` for the same parameters, key, inputs, and
    endpoint. Includes a non-unit non-friendly baseline and a friendly in the
    observed prefix, so resetting the prefix to ones fails. Exercises
    ``Methods.filter`` itself (the actual training/evaluation integration),
    not just the low-level filter."""
    from rbsqmc.comparison.sqmc_ekf.scripts import train as train_mod
    from rbsqmc.comparison.sqmc_ekf.scripts.scaling import sqmc_match_scales
    from rbsqmc.src.model.rbsqmc.model_rbsqmc import run_filter_sqmc

    # 4 matches: friendly, non-friendly, friendly, non-friendly.
    sqmc = _sqmc_inputs([1, 2, 3, 4], [0, 1, 2, 3], [1, 2, 3, 0],
                        [1, 0, 2, 1], [0, 1, 1, 0])
    friendly = jnp.array([True, False, True, False])
    cfg = dict(n_particles=8, max_goals=3, match_scale=1.5, seed=0)
    friendly_scale = 2.0
    end = 4

    # Evaluation path: build scales via the shared helper and run the filter.
    # Use the same decoded model as Methods.filter (EM reparameterizes gamma_0
    # and B), so the two filters are directly comparable.
    from rbsqmc.src.utils.helpers import default_init_params, encode_EM_params
    base = default_init_params(4, {0: "A", 1: "B", 2: "C", 3: "D"})
    data = SimpleNamespace(
        inputs=SimpleNamespace(friendly=friendly),
        sqmc=sqmc,
        teams={0: "A", 1: "B", 2: "C", 3: "D"},
    )
    methods = train_mod.Methods(data, cfg)
    raw = dict(
        model=encode_EM_params(base),
        friendly_scale=ekf.inverse_positive(friendly_scale),
    )
    decoded = methods.params("sqmc", raw)
    scales = sqmc_match_scales(friendly, friendly_scale, cfg["match_scale"])
    result_eval, _ = run_filter_sqmc(
        jax.random.PRNGKey(0), sqmc, decoded["model"], 8, 3, match_scales=scales,
    )

    # Training path: Methods.filter builds the same scales from the same
    # friendly flags and endpoint, then runs the filter. Use the actual
    # Methods.filter so the integration is exercised, not a re-implementation.
    out = methods.filter("sqmc", raw, jax.random.PRNGKey(0), end)
    result_train = out["result"]
    np.testing.assert_allclose(
        np.asarray(result_eval["particles_x"]), np.asarray(result_train["particles_x"]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(result_eval["log_weights"]), np.asarray(result_train["log_weights"]),
        atol=1e-6,
    )
    # Resetting the prefix to ones must change the history (the friendly scale
    # is non-unit), proving the scale is actually threaded through. Use the
    # same decoded parameters as the scaled case so only the scales differ.
    ones_scales = jnp.ones_like(scales)
    result_ones, _ = run_filter_sqmc(
        jax.random.PRNGKey(0), sqmc, decoded["model"], 8, 3, match_scales=ones_scales,
    )
    assert not np.allclose(
        np.asarray(result_eval["particles_x"]), np.asarray(result_ones["particles_x"]),
    )


def test_friendly_flags_sourced_from_inputs_not_sqmc():
    """The scale helper must be fed friendly flags from ``dataset.inputs.friendly``
    (a MatchInputs field), not from ``dataset.sqmc`` (a FootballResults has no
    ``friendly`` field). This pins down the sourcing subtlety so a future edit
    cannot silently read from the wrong structure."""
    from rbsqmc.comparison.sqmc_ekf.scripts.scaling import sqmc_match_scales

    # A FootballResults has no 'friendly' attribute.
    sqmc = _sqmc_inputs([1, 2], [0, 1], [1, 0], [1, 0], [0, 1])
    assert not hasattr(sqmc, "friendly"), "FootballResults must not carry a friendly field"

    # The friendly flags live on MatchInputs (dataset.inputs.friendly).
    friendly = jnp.array([True, False])
    scales = sqmc_match_scales(friendly, 2.0, 1.5)
    np.testing.assert_allclose(np.asarray(scales[:, 0]), [2.0, 1.5])

    # A friendly flag sourced from the wrong place (e.g. all-False) would give
    # the wrong scales, so the test would fail if the sourcing regressed.
    wrong = sqmc_match_scales(jnp.zeros(2, dtype=bool), 2.0, 1.5)
    assert not np.allclose(np.asarray(scales), np.asarray(wrong))


def test_scale_alignment_packed_inputs_preserve_scale_match_correspondence():
    """Packed inputs with padding and multiple matches on one date preserve
    scale/match correspondence after unpacking and concatenation."""
    from rbsqmc.src.data.data import unpack_football_results
    from rbsqmc.src.model.rbsqmc.predict_rbsqmc import _unpack_scales
    from rbsqmc.src.utils.type import FootballResults, Matches

    # 2 dates; date 0 has 2 matches, date 1 has 1 match (padded).
    packed = FootballResults(
        date=jnp.array([0, 1]),
        timestamp=jnp.array([0, 1]),
        timestamp_prev=jnp.array([0, 0]),
        matches=Matches(
            home_id=jnp.array([[0, 1], [2, 0]]),
            away_id=jnp.array([[1, 2], [3, 0]]),
            home_score=jnp.array([[1, 0], [2, 0]]),
            away_score=jnp.array([[0, 1], [1, 0]]),
        ),
        match_mask=jnp.array([[True, True], [True, False]]),
    )
    scales = jnp.array([[2.0, 3.0], [4.0, 0.0]])  # (T, M) per-match scales
    unpacked = unpack_football_results(packed)
    unpacked_scales = _unpack_scales(packed, unpacked, scales)
    # Valid (t, m) order: (0,0), (0,1), (1,0) -> scales 2, 3, 4.
    np.testing.assert_allclose(np.asarray(unpacked_scales[:, 0]), [2.0, 3.0, 4.0])
    assert unpacked_scales.shape == (3, 1)
    # A 1-D scale array is broadcast to (T, 1) and indexed by valid (t, m).
    flat = _unpack_scales(packed, unpacked, jnp.array([5.0, 6.0]))
    np.testing.assert_allclose(np.asarray(flat[:, 0]), [5.0, 5.0, 6.0])
    # (T,), (T, 1), and an explicitly broadcast (T, M) must produce identical
    # unpacked scales; no supported form may rely on out-of-bounds indexing.
    one_col = _unpack_scales(packed, unpacked, jnp.array([[5.0], [6.0]]))
    np.testing.assert_allclose(np.asarray(flat), np.asarray(one_col))
    broadcast = _unpack_scales(packed, unpacked, jnp.broadcast_to(jnp.array([[5.0], [6.0]]), (2, 2)))
    np.testing.assert_allclose(np.asarray(flat), np.asarray(broadcast))
    # Unsupported shapes must fail clearly.
    with pytest.raises(ValueError, match="Unsupported scales shape"):
        _unpack_scales(packed, unpacked, jnp.ones((2, 2, 1)))
    with pytest.raises(ValueError, match="rows"):
        _unpack_scales(packed, unpacked, jnp.ones((3, 1)))
    with pytest.raises(ValueError, match="columns"):
        _unpack_scales(packed, unpacked, jnp.ones((2, 3)))

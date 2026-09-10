"""Tests for the controlled SMC/SQMC comparison inputs."""

from datetime import datetime

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from rbsqmc.comparison.sqmc_smc.compare_smc_sqmc import (
    _comparison_output_dir,
    _format_5sf,
    _history_summary,
    _plot_gradient_norm_by_epoch,
    _plot_logz_by_epoch,
    _plot_prediction_evaluation,
    _save_method_training_artifacts,
)
from rbsqmc.src.data.data import unpack_football_results
from rbsqmc.src.utils.type import EMParams, FootballResults, Matches


def test_format_5sf_uses_significant_figures_and_handles_missing_values():
    assert _format_5sf(-15272.3707) == "-15272"
    assert _format_5sf(0.5788066616962577) == "0.57881"
    assert _format_5sf(None) == "N/A"


def test_comparison_output_dir_uses_one_minute_stamped_run_folder():
    output_dir = _comparison_output_dir(datetime(2026, 8, 27, 16, 57, 29))

    assert output_dir == "rbsqmc/outputs/compare/20260827_1657"


def test_unpack_football_results_preserves_order_and_sets_same_day_dt_to_zero():
    grouped = FootballResults(
        date=jnp.array([10, 13]),
        timestamp=jnp.array([2, 5]),
        timestamp_prev=jnp.array([0, 2]),
        matches=Matches(
            home_id=jnp.array([[0, 2], [4, 0]]),
            away_id=jnp.array([[1, 3], [5, 0]]),
            home_score=jnp.array([[1, 2], [3, 0]]),
            away_score=jnp.array([[0, 1], [2, 0]]),
        ),
        match_mask=jnp.array([[True, True], [True, False]]),
    )

    expanded = unpack_football_results(grouped)

    np.testing.assert_array_equal(expanded.date, np.array([10, 10, 13]))
    np.testing.assert_array_equal(expanded.timestamp, np.array([2, 2, 5]))
    np.testing.assert_array_equal(expanded.timestamp_prev, np.array([0, 2, 2]))
    np.testing.assert_array_equal(
        expanded.matches.home_id,
        np.array([[0], [2], [4]]),
    )
    np.testing.assert_array_equal(
        expanded.matches.away_id,
        np.array([[1], [3], [5]]),
    )
    np.testing.assert_array_equal(
        expanded.matches.home_score,
        np.array([[1], [2], [3]]),
    )
    np.testing.assert_array_equal(
        expanded.matches.away_score,
        np.array([[0], [1], [2]]),
    )
    np.testing.assert_array_equal(expanded.match_mask, np.ones((3, 1), bool))


def test_unpack_football_results_rejects_empty_inputs():
    empty = FootballResults(
        date=jnp.array([10]),
        timestamp=jnp.array([2]),
        timestamp_prev=jnp.array([0]),
        matches=Matches(
            home_id=jnp.zeros((1, 1), dtype=int),
            away_id=jnp.zeros((1, 1), dtype=int),
            home_score=jnp.zeros((1, 1), dtype=int),
            away_score=jnp.zeros((1, 1), dtype=int),
        ),
        match_mask=jnp.zeros((1, 1), dtype=bool),
    )

    with pytest.raises(ValueError, match="no valid matches"):
        unpack_football_results(empty)


def test_history_summary_reports_final_and_best_epoch_values():
    summary = _history_summary(
        train_logz=np.array([-10.0, -8.0, -7.0]),
        test_logz=np.array([-5.0, -3.0, -4.0]),
        elapsed_sec=2.5,
        train_matches=10,
        test_matches=5,
    )

    assert summary["n_epochs_completed"] == 3
    assert summary["final_train_logz"] == -7.0
    assert summary["final_train_logz_per_match"] == -0.7
    assert summary["final_test_logz"] == -4.0
    assert summary["best_test_epoch"] == 2
    assert summary["best_test_logz"] == -3.0


def test_plot_logz_by_epoch_creates_nonempty_png(tmp_path):
    history = pd.DataFrame({
        "epoch": [1, 2],
        "smc_train_logz": [-10.0, -8.0],
        "sqmc_train_logz": [-9.0, -7.0],
        "smc_test_logz": [-5.0, -4.0],
        "sqmc_test_logz": [-4.5, -3.5],
    })
    output_path = tmp_path / "logz_by_epoch.png"

    _plot_logz_by_epoch(history, str(output_path))

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_plot_gradient_norm_by_epoch_creates_nonempty_png(tmp_path):
    history = pd.DataFrame({
        "epoch": [1, 2],
        "smc_gradient_norm": [2.0, 1.0],
        "sqmc_gradient_norm": [1.5, 0.75],
    })
    output_path = tmp_path / "gradient_norm_by_epoch.png"

    _plot_gradient_norm_by_epoch(history, str(output_path))

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_method_training_artifacts_match_train_model_layout(tmp_path):
    params = EMParams(
        mean_0=jnp.zeros((2, 2)),
        gamma_0=jnp.eye(2),
        B=jnp.eye(2),
        kappa=jnp.array(0.001),
        alpha=jnp.array(0.2),
        beta=jnp.array(-4.0),
    )

    _save_method_training_artifacts(
        output_dir=str(tmp_path),
        best_params=params,
        train_logz=np.array([-10.0, -8.0]),
        test_logz=np.array([-5.0, -4.0]),
        gradient_norm=np.array([2.0, 1.0]),
        train_match_count=10,
        test_match_count=5,
    )

    expected_files = {
        "best_params.json",
        "gradient_norm_curve.json",
        "gradient_norm_curve.png",
        "logmarginal_history_train_test.json",
        "logmarginal_history_train_test.png",
    }
    assert expected_files.issubset({path.name for path in tmp_path.iterdir()})


def test_plot_prediction_evaluation_creates_nonempty_png(tmp_path):
    smc = {
        "mean_brier_score": 0.6,
        "uniform_reference_brier_score": 2.0 / 3.0,
        "exact_score_accuracy": 0.2,
        "outcome_accuracy": 0.5,
    }
    sqmc = {
        "mean_brier_score": 0.55,
        "uniform_reference_brier_score": 2.0 / 3.0,
        "exact_score_accuracy": 0.25,
        "outcome_accuracy": 0.6,
    }
    output_path = tmp_path / "prediction_evaluation.png"

    _plot_prediction_evaluation(smc, sqmc, str(output_path))

    assert output_path.is_file()
    assert output_path.stat().st_size > 0

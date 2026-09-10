"""Compare RB-SMC and RB-SQMC training over matched optimization epochs.

Both methods start from the same parameters and use the same match-level
inputs, optimizer settings, and root PRNG key. The comparison records train
and held-out test log Z after every optimization epoch.

Usage:
    python -m rbsqmc.comparison.sqmc_smc.compare_smc_sqmc
"""

import json
import os
import time
from datetime import datetime
from typing import NamedTuple

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from rbsqmc.src.data.data import (
    concat_football_results,
    get_training_data,
    unpack_football_results,
)
from rbsqmc.src.model.rbsqmc.model_rbsqmc import run_filter_sqmc
from rbsqmc.src.model.rbsmc.optimization import logmarginal_maximize, run_filter_unbiased
from rbsqmc.src.model.rbsmc.predict import (
    evaluate_match_predictions,
    run_sequential_predict,
)
from rbsqmc.src.model.rbsqmc.predict_rbsqmc import run_sequential_predict_rbsqmc
from rbsqmc.src.model.rbsmc.observe import run_observe
from rbsqmc.src.model.rbsqmc.train_model_rbsqmc import logmarginal_maximize_sqmc
from rbsqmc.src.utils.graphic import (
    plot_all,
    plot_gradient_norm_curve,
    plot_logmarginal_history_train_test,
)
from rbsqmc.src.utils.helpers import (
    default_init_params,
    build_match_predictions,
    resolve_teams,
    save_match_predictions,
    save_params,
)
from rbsqmc.src.utils.type import RBPFState

def _comparison_output_dir(now: datetime | None = None) -> str:
    """Return the single minute-stamped directory for a comparison run."""
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M")
    return os.path.join("rbsqmc", "outputs", "compare", timestamp)

output_dir = _comparison_output_dir()

cfg = {
    "training_start_date": "1980-01-01",
    "test_start_date": "2024-01-01",
    "prediction_start_date": "2026-06-11",
    # A power of two preserves the balance properties of the Sobol net.
    "n_particles": 256,
    "max_goals": 8,
    "seed": 0,
    "n_epochs": 100,
    "learning_rate": 0.05,
    # Independent filter replicas averaged for each epoch's gradient.
    "n_reps": 20,
    # Keep both histories aligned at exactly n_epochs for comparison.
    "patience": None,
    "include_friendly": True,
    "teams": "worldcup2026",
    "observation_step": "match",
    "output_dir": output_dir,
}

class SQMCFilteredStates(NamedTuple):
    """Adapter exposing SQMC arrays through the interface used by plot_all."""

    particles: RBPFState
    log_weights: jax.Array
    log_normalizing_constant: jax.Array


def _format_5sf(value: float | None) -> str:
    """Format an optional summary value to five significant figures."""
    return "N/A" if value is None else f"{value:.5g}"


def _plot_logz_by_epoch(history: pd.DataFrame, output_path: str) -> None:
    """Plot matched SMC/SQMC train and test log Z trajectories."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    method_styles = {
        "SMC": {"color": "steelblue", "linestyle": "-", "marker": "o"},
        "SQMC": {"color": "darkred", "linestyle": "--", "marker": "s"},
    }

    for axis, split in zip(axes, ("train", "test")):
        for method, style in method_styles.items():
            axis.plot(
                history["epoch"],
                history[f"{method.lower()}_{split}_logz"],
                label=method,
                linewidth=1.8,
                markersize=4,
                **style,
            )
        axis.set_title(f"{split.title()} Log Marginal Likelihood per Epoch")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(f"{split.title()} Log Marginal Likelihood (logZ)")
        axis.grid(True, alpha=0.3)
        axis.legend()
        axis.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        axis.tick_params(axis="x", rotation=45)

    fig.suptitle("RB-SMC and RB-SQMC Log Marginal Likelihood per Epoch")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_gradient_norm_by_epoch(
    history: pd.DataFrame,
    output_path: str,
) -> None:
    """Plot matched SMC/SQMC gradient norms across optimization epochs."""
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.plot(
        history["epoch"],
        history["smc_gradient_norm"],
        label="SMC",
        color="steelblue",
        linestyle="-",
        marker="o",
        linewidth=1.8,
    )
    axis.plot(
        history["epoch"],
        history["sqmc_gradient_norm"],
        label="SQMC",
        color="darkred",
        linestyle="--",
        marker="s",
        linewidth=1.8,
    )
    axis.set_title("Gradient Norm per Epoch")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Global gradient norm")
    axis.grid(True, alpha=0.3)
    axis.legend()
    axis.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    axis.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_prediction_evaluation(
    smc_evaluation: dict,
    sqmc_evaluation: dict,
    output_path: str,
) -> None:
    """Plot Brier scores and prediction accuracies for both methods."""
    fig, (brier_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 5.5))
    methods = ["SMC", "SQMC"]
    colors = ["steelblue", "darkred"]
    evaluations = [smc_evaluation, sqmc_evaluation]

    brier_values = [
        np.nan if item["mean_brier_score"] is None else item["mean_brier_score"]
        for item in evaluations
    ]
    brier_axis.bar(methods, brier_values, color=colors, width=0.6)
    reference = smc_evaluation["uniform_reference_brier_score"]
    brier_axis.axhline(
        reference,
        color="dimgray",
        linestyle="--",
        linewidth=1.8,
        label=f"Uniform reference ({reference:.3f})",
    )
    brier_axis.set_title("Three-outcome Brier Score")
    brier_axis.set_ylabel("Mean Brier score (lower is better)")
    brier_axis.grid(True, axis="y", alpha=0.3)
    brier_axis.legend()

    x = np.arange(len(methods))
    width = 0.34
    exact_accuracy = [
        np.nan if item["exact_score_accuracy"] is None else item["exact_score_accuracy"]
        for item in evaluations
    ]
    outcome_accuracy = [
        np.nan if item["outcome_accuracy"] is None else item["outcome_accuracy"]
        for item in evaluations
    ]
    accuracy_axis.bar(
        x - width / 2,
        exact_accuracy,
        width,
        color="slategray",
        label="Exact score",
    )
    accuracy_axis.bar(
        x + width / 2,
        outcome_accuracy,
        width,
        color="seagreen",
        label="Outcome",
    )
    accuracy_axis.set_xticks(x, methods)
    accuracy_axis.set_ylim(0.0, 1.0)
    accuracy_axis.set_title("Prediction Accuracy")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.grid(True, axis="y", alpha=0.3)
    accuracy_axis.legend()

    fig.suptitle("RB-SMC and RB-SQMC Prediction Evaluation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _history_summary(train_logz, test_logz, elapsed_sec, train_matches, test_matches):
    """Return compact final/best diagnostics for one optimization run."""
    train_logz = np.asarray(train_logz, dtype=float)
    test_logz = np.asarray(test_logz, dtype=float)
    finite_test = np.isfinite(test_logz)
    best_test_index = int(np.nanargmax(test_logz)) if finite_test.any() else -1
    best_test_epoch = best_test_index + 1 if best_test_index >= 0 else -1
    best_test_logz = (
        float(test_logz[best_test_index])
        if best_test_index >= 0
        else float("nan")
    )

    return {
        "n_epochs_completed": int(train_logz.size),
        "elapsed_sec": float(elapsed_sec),
        "final_train_logz": float(train_logz[-1]),
        "final_train_logz_per_match": float(train_logz[-1] / train_matches),
        "final_test_logz": float(test_logz[-1]),
        "final_test_logz_per_match": float(test_logz[-1] / test_matches),
        "best_test_epoch": best_test_epoch,
        "best_test_logz": best_test_logz,
        "best_test_logz_per_match": float(best_test_logz / test_matches),
        "train_logz_history": train_logz.tolist(),
        "test_logz_history": test_logz.tolist(),
    }


def _save_method_training_artifacts(
    output_dir,
    best_params,
    train_logz,
    test_logz,
    gradient_norm,
    train_match_count,
    test_match_count,
) -> None:
    """Save the same core optimization artifacts as train_model.py."""
    os.makedirs(output_dir, exist_ok=True)
    figure, _ = plot_logmarginal_history_train_test(
        train_logz_history=train_logz,
        train_match_count=train_match_count,
        test_logz_history=test_logz,
        test_match_count=test_match_count,
        save_path=os.path.join(
            output_dir,
            "logmarginal_history_train_test.png",
        ),
    )
    plt.close(figure)
    figure, _ = plot_gradient_norm_curve(
        grad_norm_history=gradient_norm,
        save_path=os.path.join(output_dir, "gradient_norm_curve.png"),
    )
    plt.close(figure)
    save_params(best_params, os.path.join(output_dir, "best_params.json"))


def _save_final_filter_artifacts(
    method,
    key,
    model_inputs,
    params,
    n_particles,
    max_goals,
    team_id_to_name,
    timestamps,
    output_dir,
) -> None:
    """Run the selected best checkpoint and save plot_all diagnostics."""
    if method == "smc":
        filtered_states, augmented_inputs = run_filter_unbiased(
            key=key,
            model_inputs=model_inputs,
            params=params,
            n_particles=n_particles,
            max_goals=max_goals,
        )
    elif method == "sqmc":
        result, augmented_inputs = run_filter_sqmc(
            key=key,
            model_inputs=model_inputs,
            params=params,
            n_particles=n_particles,
            max_goals=max_goals,
        )
        filtered_states = SQMCFilteredStates(
            particles=RBPFState(x=result["particles_x"]),
            log_weights=result["log_weights"],
            log_normalizing_constant=result["log_normalizing_constant"],
        )
    else:
        raise ValueError(f"Unknown filtering method: {method}")

    final_filter_dir = os.path.join(output_dir, "final_filter")
    os.makedirs(final_filter_dir, exist_ok=True)
    plot_all(
        filtered_states=filtered_states,
        augmented_results=augmented_inputs,
        team_id_to_name=team_id_to_name,
        top_n=10,
        save_path=final_filter_dir,
        timestamps=timestamps,
        params=params,
    )


def _save_prediction_artifacts(
    method,
    key,
    observed_inputs,
    prediction_inputs,
    params,
    n_particles,
    max_goals,
    team_id_to_name,
    output_dir,
) -> dict:
    """Run sequential prediction, save forecasts, and return evaluation."""
    if method == "smc":
        grids, log_probabilities, _ = run_sequential_predict(
            key=key,
            observed_inputs=observed_inputs,
            prediction_inputs=prediction_inputs,
            params=params,
            n_particles=n_particles,
            max_goals=max_goals,
        )
    elif method == "sqmc":
        grids, log_probabilities, _ = run_sequential_predict_rbsqmc(
            key=key,
            observed_inputs=observed_inputs,
            prediction_inputs=prediction_inputs,
            params=params,
            n_particles=n_particles,
            max_goals=max_goals,
        )
    else:
        raise ValueError(f"Unknown prediction method: {method}")

    grids.block_until_ready()
    predictions = build_match_predictions(
        all_grids=grids,
        all_logp_actual=log_probabilities,
        prediction_inputs=prediction_inputs,
        team_id_to_name=team_id_to_name,
        max_goals=max_goals,
    )
    evaluation = evaluate_match_predictions(predictions)
    prediction_dir = os.path.join(output_dir, "prediction")
    save_match_predictions(
        predictions,
        save_dir=prediction_dir,
        max_goals=max_goals,
    )
    with open(os.path.join(prediction_dir, "evaluation.json"), "w") as f:
        json.dump(evaluation, f, indent=2)
    # Post-prediction observe artifacts: final_filter_after_prediction/ plus
    # observe_team_states*.png/json written into prediction_dir, mirroring
    # train_model.py's run_observe step. Method-aware so SQMC uses its own
    # forward filter.
    run_observe(
        cfg=cfg,
        params=params,
        output_dir=prediction_dir,
        method=method,
    )
    return evaluation


def main():
    # Do not silently mix two comparison runs started in the same minute.
    os.makedirs(output_dir, exist_ok=False)
    print(f"Output directory: {output_dir}")

    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    key = jax.random.PRNGKey(cfg["seed"])
    teams_only = resolve_teams(cfg)
    (train_df, test_df, prediction_df), (
        train_model_inputs,
        test_model_inputs,
        prediction_model_inputs,
    ), team_id_to_name = get_training_data(
        train_start_date=cfg["training_start_date"],
        test_start_date=cfg["test_start_date"],
        prediction_start_date=cfg["prediction_start_date"],
        max_goals=cfg["max_goals"],
        include_friendly=cfg["include_friendly"],
        teams_only=teams_only,
    )

    train_date_count = len(train_df)
    test_date_count = len(test_df)
    train_model_inputs = unpack_football_results(train_model_inputs)
    test_model_inputs = unpack_football_results(test_model_inputs)
    prediction_model_inputs = unpack_football_results(prediction_model_inputs)
    train_match_count = int(train_model_inputs.match_mask.sum())
    test_match_count = int(test_model_inputs.match_mask.sum())
    prediction_match_count = int(prediction_model_inputs.match_mask.sum())
    num_teams = len(team_id_to_name)

    print(f"Teams: {num_teams}")
    print(
        f"Train: {train_match_count} matches on {train_date_count} dates; "
        f"test: {test_match_count} matches on {test_date_count} dates."
    )
    print(
        f"Training both filters for {cfg['n_epochs']} epochs with "
        f"{cfg['n_reps']} replicas per gradient."
    )

    initial_params = default_init_params(
        num_teams=num_teams,
        team_id_to_name=team_id_to_name,
    )
    optimizer_kwargs = {
        "train_model_inputs": train_model_inputs,
        "test_model_inputs": test_model_inputs,
        "params": initial_params,
        "n_particles": cfg["n_particles"],
        "max_goals": cfg["max_goals"],
        "n_epochs": cfg["n_epochs"],
        "learning_rate": cfg["learning_rate"],
        "n_reps": cfg["n_reps"],
        "patience": cfg["patience"],
    }

    print(f"\n{'=' * 60}\nTraining RB-SMC\n{'=' * 60}")
    t0 = time.perf_counter()
    smc_best_params, smc_train, smc_test, smc_grad = logmarginal_maximize(
        key=key,
        **optimizer_kwargs,
    )
    smc_elapsed = time.perf_counter() - t0

    print(f"\n{'=' * 60}\nTraining RB-SQMC\n{'=' * 60}")
    t0 = time.perf_counter()
    sqmc_best_params, sqmc_train, sqmc_test, sqmc_grad = logmarginal_maximize_sqmc(
        key=key,
        **optimizer_kwargs,
    )
    sqmc_elapsed = time.perf_counter() - t0

    smc_train = np.asarray(smc_train, dtype=float)
    smc_test = np.asarray(smc_test, dtype=float)
    smc_grad = np.asarray(smc_grad, dtype=float)
    sqmc_train = np.asarray(sqmc_train, dtype=float)
    sqmc_test = np.asarray(sqmc_test, dtype=float)
    sqmc_grad = np.asarray(sqmc_grad, dtype=float)

    lengths = {smc_train.size, smc_test.size, sqmc_train.size, sqmc_test.size}
    if lengths != {cfg["n_epochs"]}:
        raise RuntimeError(
            "The comparison requires both methods to complete exactly "
            f"{cfg['n_epochs']} epochs; observed history lengths {sorted(lengths)}."
        )

    history = pd.DataFrame({
        "epoch": np.arange(1, cfg["n_epochs"] + 1),
        "smc_train_logz": smc_train,
        "sqmc_train_logz": sqmc_train,
        "smc_test_logz": smc_test,
        "sqmc_test_logz": sqmc_test,
        "smc_train_logz_per_match": smc_train / train_match_count,
        "sqmc_train_logz_per_match": sqmc_train / train_match_count,
        "smc_test_logz_per_match": smc_test / test_match_count,
        "sqmc_test_logz_per_match": sqmc_test / test_match_count,
        "smc_gradient_norm": smc_grad,
        "sqmc_gradient_norm": sqmc_grad,
    })
    history_path = os.path.join(output_dir, "logz_by_epoch.csv")
    history.to_csv(history_path, index=False)
    plot_path = os.path.join(output_dir, "logz_by_epoch.png")
    _plot_logz_by_epoch(history, plot_path)
    gradient_plot_path = os.path.join(output_dir, "gradient_norm_by_epoch.png")
    _plot_gradient_norm_by_epoch(history, gradient_plot_path)
    with open(os.path.join(output_dir, "logz_by_epoch.json"), "w") as f:
        json.dump(
            history[[
                "epoch",
                "smc_train_logz",
                "sqmc_train_logz",
                "smc_test_logz",
                "sqmc_test_logz",
            ]].to_dict(orient="list"),
            f,
            indent=2,
        )
    with open(os.path.join(output_dir, "gradient_norm_by_epoch.json"), "w") as f:
        json.dump(
            history[[
                "epoch",
                "smc_gradient_norm",
                "sqmc_gradient_norm",
            ]].to_dict(orient="list"),
            f,
            indent=2,
        )

    smc_dir = os.path.join(output_dir, "smc")
    sqmc_dir = os.path.join(output_dir, "sqmc")
    os.makedirs(smc_dir, exist_ok=True)
    os.makedirs(sqmc_dir, exist_ok=True)
    for method, method_dir in (("smc", smc_dir), ("sqmc", sqmc_dir)):
        with open(os.path.join(method_dir, "run_config.json"), "w") as f:
            json.dump({**cfg, "method": method}, f, indent=2)
    _save_method_training_artifacts(
        smc_dir,
        smc_best_params,
        smc_train,
        smc_test,
        smc_grad,
        train_match_count,
        test_match_count,
    )
    _save_method_training_artifacts(
        sqmc_dir,
        sqmc_best_params,
        sqmc_train,
        sqmc_test,
        sqmc_grad,
        train_match_count,
        test_match_count,
    )

    smc_summary = _history_summary(
        smc_train, smc_test, smc_elapsed, train_match_count, test_match_count
    )
    sqmc_summary = _history_summary(
        sqmc_train, sqmc_test, sqmc_elapsed, train_match_count, test_match_count
    )
    summary = {
        "config": cfg,
        "num_teams": num_teams,
        "train_matches": train_match_count,
        "test_matches": test_match_count,
        "prediction_matches": prediction_match_count,
        "train_dates": train_date_count,
        "test_dates": test_date_count,
        "prediction_dates": len(prediction_df),
        "steps_per_match": 1,
        "timings_include_jit_compilation": True,
        "smc": smc_summary,
        "sqmc": sqmc_summary,
        "comparison": {
            "final_train_logz_diff": float(sqmc_train[-1] - smc_train[-1]),
            "final_test_logz_diff": float(sqmc_test[-1] - smc_test[-1]),
            "best_test_logz_diff": float(
                sqmc_summary["best_test_logz"]
                - smc_summary["best_test_logz"]
            ),
        },
    }
    observed_inputs = concat_football_results(
        train_model_inputs,
        test_model_inputs,
    )
    observed_dates = pd.to_datetime(
        np.asarray(observed_inputs.date),
        unit="D",
        origin="unix",
    ).to_numpy()
    _, final_filter_key = jax.random.split(key)
    print(f"\n{'=' * 60}\nSaving RB-SMC final-filter artifacts\n{'=' * 60}")
    _save_final_filter_artifacts(
        method="smc",
        key=final_filter_key,
        model_inputs=observed_inputs,
        params=smc_best_params,
        n_particles=cfg["n_particles"],
        max_goals=cfg["max_goals"],
        team_id_to_name=team_id_to_name,
        timestamps=observed_dates,
        output_dir=smc_dir,
    )
    print(f"\n{'=' * 60}\nSaving RB-SQMC final-filter artifacts\n{'=' * 60}")
    _save_final_filter_artifacts(
        method="sqmc",
        key=final_filter_key,
        model_inputs=observed_inputs,
        params=sqmc_best_params,
        n_particles=cfg["n_particles"],
        max_goals=cfg["max_goals"],
        team_id_to_name=team_id_to_name,
        timestamps=observed_dates,
        output_dir=sqmc_dir,
    )

    _, prediction_key = jax.random.split(final_filter_key)
    print(f"\n{'=' * 60}\nRunning RB-SMC predictions\n{'=' * 60}")
    smc_prediction_evaluation = _save_prediction_artifacts(
        method="smc",
        key=prediction_key,
        observed_inputs=observed_inputs,
        prediction_inputs=prediction_model_inputs,
        params=smc_best_params,
        n_particles=cfg["n_particles"],
        max_goals=cfg["max_goals"],
        team_id_to_name=team_id_to_name,
        output_dir=smc_dir,
    )
    print(f"\n{'=' * 60}\nRunning RB-SQMC predictions\n{'=' * 60}")
    sqmc_prediction_evaluation = _save_prediction_artifacts(
        method="sqmc",
        key=prediction_key,
        observed_inputs=observed_inputs,
        prediction_inputs=prediction_model_inputs,
        params=sqmc_best_params,
        n_particles=cfg["n_particles"],
        max_goals=cfg["max_goals"],
        team_id_to_name=team_id_to_name,
        output_dir=sqmc_dir,
    )
    summary["smc"]["prediction_evaluation"] = smc_prediction_evaluation
    summary["sqmc"]["prediction_evaluation"] = sqmc_prediction_evaluation

    evaluation_metrics = [
        "n_predictions",
        "n_scored",
        "mean_brier_score",
        "uniform_reference_brier_score",
        "brier_skill_score_vs_uniform",
        "total_log_likelihood",
        "mean_log_likelihood",
        "exact_score_accuracy",
        "outcome_accuracy",
    ]
    evaluation_path = os.path.join(output_dir, "prediction_evaluation.csv")
    pd.DataFrame({
        "metric": evaluation_metrics,
        "smc": [smc_prediction_evaluation[key] for key in evaluation_metrics],
        "sqmc": [sqmc_prediction_evaluation[key] for key in evaluation_metrics],
    }).to_csv(evaluation_path, index=False)
    evaluation_plot_path = os.path.join(output_dir, "prediction_evaluation.png")
    _plot_prediction_evaluation(
        smc_prediction_evaluation,
        sqmc_prediction_evaluation,
        evaluation_plot_path,
    )

    results_path = os.path.join(output_dir, "comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    uniform_brier = smc_prediction_evaluation["uniform_reference_brier_score"]
    smc_brier = smc_prediction_evaluation["mean_brier_score"]
    sqmc_brier = sqmc_prediction_evaluation["mean_brier_score"]
    brier_diff = (
        sqmc_brier - smc_brier
        if smc_brier is not None and sqmc_brier is not None
        else None
    )
    summary_rows = [
        (
            "Final train logZ",
            smc_train[-1],
            sqmc_train[-1],
            sqmc_train[-1] - smc_train[-1],
        ),
        (
            "Final test logZ",
            smc_test[-1],
            sqmc_test[-1],
            sqmc_test[-1] - smc_test[-1],
        ),
        (
            "Best test logZ",
            smc_summary["best_test_logz"],
            sqmc_summary["best_test_logz"],
            summary["comparison"]["best_test_logz_diff"],
        ),
        (
            "Training time (s)",
            smc_elapsed,
            sqmc_elapsed,
            sqmc_elapsed - smc_elapsed,
        ),
        (
            f"Prediction Brier score (baseline {uniform_brier:.2f})",
            smc_brier,
            sqmc_brier,
            brier_diff,
        ),
    ]

    metric_width = 40
    table_width = metric_width + 3 * 16
    print(f"\n{'=' * table_width}\nSUMMARY\n{'=' * table_width}")
    print(
        f"{'Metric':<{metric_width}} {'SMC':>15} "
        f"{'SQMC':>15} {'SQMC-SMC':>15}"
    )
    print("-" * table_width)
    for metric, smc_value, sqmc_value, difference in summary_rows:
        print(
            f"{metric:<{metric_width}} "
            f"{_format_5sf(smc_value):>15} "
            f"{_format_5sf(sqmc_value):>15} "
            f"{_format_5sf(difference):>15}"
        )
    print(f"\nPer-epoch history saved to {history_path}")
    print(f"Per-epoch plot saved to {plot_path}")
    print(f"Gradient-norm plot saved to {gradient_plot_path}")
    print(f"Prediction evaluation saved to {evaluation_path}")
    print(f"Prediction evaluation plot saved to {evaluation_plot_path}")
    print(f"Summary saved to {results_path}")


if __name__ == "__main__":
    main()

"""Sweep over ``training_start_date`` values and compare fit quality.

For each combination of **start date x prior setting x ``n_particles`` x
``n_reps``**, this runs the log-marginal optimization with the **test split
used as the validation signal** (so the returned ``best_params`` is the
checkpoint with the highest held-out test logZ, not the overfit final params).
It records per-config metrics into a single comparison summary (CSV + JSON) so
you can see which combination generalizes best on the held-out split.

Sweep dimensions (each defaults to a list, mirroring ``DEFAULT_START_DATES``):
  - ``--start-dates``   (default ``DEFAULT_START_DATES``)
  - ``--prior``         (none / iw / both; default ``both``)
  - ``--n_particles``   (default ``DEFAULT_N_PARTICLES`` = 250 500)
  - ``--n_reps``        (default ``DEFAULT_N_REPS`` = 20 30)

The total number of runs is the cross-product of these dimensions. Each run
writes to its own subfolder (e.g. ``19500101_noprior_N250_rep30``).

By default each start date is run **twice**: once with the inverse-Wishart
prior on ``gamma_0`` and once without (``--prior both``). Use ``--prior none``
or ``--prior iw`` to run only one variant.

Prediction runs by default (``--predict``): after optimization, the filter +
sequential prediction is run on the upcoming fixtures and the average
prediction log-likelihood is recorded. Pass ``--no-predict`` to skip it.

``test_start_date`` (default 2024-01-01) and ``prediction_start_date`` (default
2026-06-11) mirror ``rbsqmc/src/model/train_model.py``.

Results are written to ``rbsqmc/outputs/parameter_sweep/YYYYMMDD_HHMM/`` by
default (override with ``--output_root``).

Usage:
    python -m rbsqmc.comparison.sqmc_smc.sweep_start_dates \
        --start-dates 1950-01-01 1970-01-01 1990-01-01 \
        --n_particles 250 500 --n_reps 20 30 --n_epochs 100 --patience 15 \
        --test_start_date 2024-01-01 --prediction_start_date 2026-06-11
"""

import argparse
import json
import logging
import os
from datetime import datetime

import jax
import numpy as np
import pandas as pd

from rbsqmc.src.data.data import get_training_data, concat_football_results
from rbsqmc.src.model.rbsmc.optimization import logmarginal_maximize, run_filter_unbiased
from rbsqmc.src.utils.helpers import default_init_params, resolve_teams, save_params

logger = logging.getLogger("rbsqmc.sweep")


def setup_logging(log_path: str) -> None:
    """Configure logging to write to ``log_path`` (and the console).

    Must be called once at the start of ``main``, before any runs begin, so the
    entire sweep (including early setup lines) is captured in the log.
    """
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)


def build_search_space(args) -> dict:
    """Serialize the sweep search space into a JSON-serialisable dict.

    Includes the start dates, prior variants, n_particles and n_reps lists, and
    the fixed (non-swept) config so the exact grid is reproducible from the log.
    """
    variants = prior_variants(args)
    return {
        "start_dates": list(args.start_dates),
        "prior": args.prior,
        "prior_variants": [
            {"tag": tag, "gamma_0_prior_params": gp} for (tag, gp) in variants
        ],
        "n_particles": list(args.n_particles),
        "n_reps": list(args.n_reps),
        "test_start_date": args.test_start_date,
        "prediction_start_date": args.prediction_start_date,
        "max_goals": args.max_goals,
        "seed": args.seed,
        "n_epochs": args.n_epochs,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "include_friendly": args.include_friendly,
        "teams": args.teams,
        "download": args.download,
        "predict": args.predict,
    }


DEFAULT_START_DATES = [
    "1950-01-01",
    "1960-01-01",
    "1970-01-01",
    "1980-01-01",
    "1990-01-01",
    "2000-01-01",
    "2010-01-01",
]

# Sweep over n_particles (particle count N) values.
DEFAULT_N_PARTICLES = [250, 500]

# Sweep over n_reps (number of filter replicas averaged for the gradient).
DEFAULT_N_REPS = [20, 30]


def build_config(args) -> dict:
    """Construct the per-run config dict shared by all sweep iterations.

    ``n_particles`` / ``n_reps`` are not stored here because they are swept per
    run (each combination of start date x prior x n_particles x n_reps is one
    run). They are injected into ``cfg`` inside ``run_one``.
    """
    cfg = {
        "training_start_date": None,  # filled per iteration
        "test_start_date": args.test_start_date,
        "prediction_start_date": args.prediction_start_date,
        "max_goals": args.max_goals,
        "seed": args.seed,
        "n_epochs": args.n_epochs,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "gamma_0_prior_params": None,  # set per prior variant in run_one
        "include_friendly": args.include_friendly,
        "teams": args.teams,
        "download": args.download,
    }
    return cfg


def prior_variants(args):
    """Return a list of (tag, gamma_0_prior_params) to sweep.

    ``args.prior`` is one of ``none``, ``iw``, or ``both``:
      - none : only without the inverse-Wishart prior (gamma_0_prior_params=None)
      - iw   : only with the inverse-Wishart prior
      - both : run both (default), so each start date is compared with and without.
    """
    with_prior = {
        "scale": None,  # defaults to the initial gamma_0 matrix inside logmarginal_maximize
        "dof": args.gamma_dof,
        "strength": args.gamma_strength,
    }
    variants = []
    if args.prior in ("none", "both"):
        variants.append(("noprior", None))
    if args.prior in ("iw", "both"):
        variants.append(("iwprior", with_prior))
    return variants


def run_one(cfg: dict, start_date: str, prior_tag: str, gamma_0_prior_params,
            n_particles: int, n_reps: int,
            output_root: str, run_predict: bool, log_path: str) -> dict:
    """Run the optimization (and optional prediction) for one hyperparameter combo."""
    cfg = dict(cfg)
    cfg["training_start_date"] = start_date
    # Deep-copy the prior params so `logmarginal_maximize`'s in-place mutation
    # (it fills scale=None with the initial gamma_0 JAX array) does not leak
    # into subsequent runs / the JSON-serialisable config written here.
    cfg["gamma_0_prior_params"] = (
        dict(gamma_0_prior_params) if gamma_0_prior_params is not None else None
    )
    cfg["n_particles"] = n_particles
    cfg["n_reps"] = n_reps
    run_dir = os.path.join(
        output_root,
        f"{start_date.replace('-', '')}_{prior_tag}_N{n_particles}_rep{n_reps}",
    )
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    key = jax.random.PRNGKey(cfg["seed"])
    teams_only = resolve_teams(cfg)
    (train_df, test_df, pred_df), (
        train_inputs,
        test_inputs,
        pred_inputs,
    ), team_id_to_name = get_training_data(
        train_start_date=start_date,
        test_start_date=cfg["test_start_date"],
        prediction_start_date=cfg["prediction_start_date"],
        max_goals=cfg["max_goals"],
        include_friendly=cfg["include_friendly"],
        teams_only=teams_only,
        download=cfg["download"],
    )
    num_teams = len(team_id_to_name)
    params = default_init_params(num_teams=num_teams, team_id_to_name=team_id_to_name)

    key, opt_key = jax.random.split(key, 2)
    best_params, train_history, test_history, grad_history = logmarginal_maximize(
        key=opt_key,
        train_model_inputs=train_inputs,
        test_model_inputs=test_inputs,  # held-out used as the validation signal
        params=params,
        n_particles=cfg["n_particles"],
        max_goals=cfg["max_goals"],
        n_epochs=cfg["n_epochs"],
        learning_rate=cfg["learning_rate"],
        n_reps=cfg["n_reps"],
        gamma_0_prior_params=cfg["gamma_0_prior_params"],
        patience=cfg["patience"],
    )

    train_history = [float(v) for v in train_history]
    test_history = [float(v) for v in test_history]
    grad_history = [float(v) for v in grad_history]

    best_test_logz = max(test_history) if test_history else float("nan")
    best_test_epoch = int(np.argmax(test_history)) if test_history else -1
    best_train_logz = max(train_history) if train_history else float("nan")

    # Save the test-selected checkpoint and the raw curves for inspection.
    save_params(params=best_params, path=os.path.join(run_dir, "best_params.json"))
    with open(os.path.join(run_dir, "logmarginal_history.json"), "w") as f:
        json.dump(
            {
                "epoch": list(range(len(train_history))),
                "train_logz": train_history,
                "test_logz": test_history,
                "gradient_norm": grad_history,
                "train_match_count": int(train_inputs.match_mask.sum()),
                "test_match_count": int(test_inputs.match_mask.sum()),
            },
            f,
            indent=2,
        )

    result = {
        "training_start_date": start_date,
        "prior": prior_tag,
        "gamma_0_prior_params": gamma_0_prior_params,
        "n_particles": n_particles,
        "n_reps": n_reps,
        "train_matches": int(train_inputs.match_mask.sum()),
        "test_matches": int(test_inputs.match_mask.sum()),
        "prediction_matches": int(pred_inputs.match_mask.sum()),
        "best_train_logz": best_train_logz,
        "best_test_logz": best_test_logz,
        "best_test_logz_per_match": (
            best_test_logz / int(test_inputs.match_mask.sum())
            if int(test_inputs.match_mask.sum()) > 0
            else float("nan")
        ),
        "best_test_epoch": best_test_epoch,
        "final_gradient_norm": grad_history[-1] if grad_history else float("nan"),
        "n_epochs_actual": len(train_history),
    }

    if run_predict:
        result.update(
            _predict(cfg, best_params, train_inputs, test_inputs, pred_inputs, team_id_to_name, run_dir, log_path)
        )

    logger.info(
        f"[sweep] {start_date} [{prior_tag}] N={n_particles} reps={n_reps}: "
        f"best_test_logZ={best_test_logz:.2f} "
        f"({result['best_test_logz_per_match']:.4f}/match @ ep {best_test_epoch})"
    )
    return result


def _predict(cfg, params, train_inputs, test_inputs, pred_inputs, team_id_to_name, run_dir, log_path) -> dict:
    """Sequential prediction on the upcoming-match split (left to last stage)."""
    from rbsqmc.src.model.rbsmc.predict import run_sequential_predict
    from rbsqmc.src.utils.helpers import build_match_predictions, save_match_predictions

    key = jax.random.PRNGKey(cfg["seed"] + 1)
    observed_inputs = concat_football_results(train_inputs, test_inputs)
    pred_grids, pred_logprobs, _daily = run_sequential_predict(
        key=key,
        observed_inputs=observed_inputs,
        prediction_inputs=pred_inputs,
        params=params,
        n_particles=cfg["n_particles"],
        max_goals=cfg["max_goals"],
    )

    predictions = build_match_predictions(
        all_grids=pred_grids,
        all_logp_actual=pred_logprobs,
        prediction_inputs=pred_inputs,
        team_id_to_name=team_id_to_name,
        max_goals=cfg["max_goals"],
    )
    pred_dir = os.path.join(run_dir, "prediction")
    save_match_predictions(predictions, save_dir=pred_dir, max_goals=cfg["max_goals"])

    # Aggregate prediction performance: average per-match log-likelihood and
    # winner accuracy (predicted outcome vs actual result).
    n = len(predictions)
    avg_loglik = (
        float(np.mean([m["log_likelihood"] for m in predictions])) if n else float("nan")
    )
    correct = 0
    for m in predictions:
        ah, aa = m["actual_home_score"], m["actual_away_score"]
        actual = "H" if ah > aa else ("A" if aa > ah else "D")
        probs = {
            "H": float(m["prob_home_win"]),
            "D": float(m["prob_draw"]),
            "A": float(m["prob_away_win"]),
        }
        predicted = max(probs, key=probs.__getitem__)
        if predicted == actual:
            correct += 1
    winner_acc = correct / n if n else float("nan")

    return {
        "prediction_avg_loglik": avg_loglik,
        "prediction_winner_accuracy": winner_acc,
        "prediction_correct": correct,
        "prediction_matches_evaluated": n,
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep training_start_date and compare fit.")
    parser.add_argument("--start-dates", nargs="+", default=DEFAULT_START_DATES,
                        help="training_start_date values to sweep (YYYY-MM-DD).")
    parser.add_argument("--test_start_date", default="2024-01-01")
    parser.add_argument("--prediction_start_date", default="2026-06-11")
    parser.add_argument("--n_particles", nargs="+", type=int, default=DEFAULT_N_PARTICLES,
                        help="Particle count N value(s) to sweep (default: %(default)s).")
    parser.add_argument("--max_goals", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--n_reps", nargs="+", type=int, default=DEFAULT_N_REPS,
                        help="Number of filter replicas per gradient value(s) to sweep (default: %(default)s).")
    parser.add_argument("--patience", type=int, default=None,
                        help="Early-stop after this many epochs without held-out test improvement.")
    parser.add_argument("--include_friendly", type=lambda v: v.lower() in ("1", "true", "yes"), default=True)
    parser.add_argument("--teams", default="worldcup2026")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--predict", action="store_true", default=True,
                        help="Also run filter + sequential prediction for each start date (default: on).")
    parser.add_argument("--no-predict", dest="predict", action="store_false",
                        help="Skip the sequential prediction phase (optimization only).")
    parser.add_argument("--prior", choices=["none", "iw", "both"], default="both",
                        help="Which gamma_0 prior variants to run per start date: none, iw, or both (default: both).")
    parser.add_argument("--gamma_scale", default=None,
                        help="Optional gamma_0 prior scale; None (default) uses the initial gamma_0 matrix.")
    parser.add_argument("--gamma_dof", type=float, default=5.0)
    parser.add_argument("--gamma_strength", type=float, default=1.0)
    parser.add_argument("--output_root", default=None)
    args = parser.parse_args()

    output_root = args.output_root or os.path.join(
        "rbsqmc", "outputs", "parameter_sweep",
        datetime.now().strftime("%Y%m%d_%H%M"),
    )
    os.makedirs(output_root, exist_ok=True)

    # Set up root-level logging FIRST (before any runs) so the entire sweep is
    # captured in a single log at the YYYYMMDD_HHMM level.
    log_path = os.path.join(output_root, "sweep.log")
    setup_logging(log_path)
    logger.info(f"Sweep output root: {os.path.abspath(output_root)}")

    cfg = build_config(args)
    cfg["training_start_date"] = None
    variants = prior_variants(args)

    # Save the full search space (all swept dimensions + fixed config) as JSON
    # at the root level, written at the start.
    search_space = build_search_space(args)
    search_space_path = os.path.join(output_root, "search_space.json")
    with open(search_space_path, "w") as f:
        json.dump(search_space, f, indent=2)
    logger.info(f"Saved search space to {os.path.abspath(search_space_path)}")

    with open(os.path.join(output_root, "sweep_config.json"), "w") as f:
        json.dump(
            {
                "start_dates": args.start_dates,
                "prior": args.prior,
                "gamma_0_prior_variants": variants,
                "test_start_date": args.test_start_date,
                "prediction_start_date": args.prediction_start_date,
                **cfg,
            },
            f,
            indent=2,
        )

    # Each start date x prior variant x n_particles x n_reps is one run.
    runs = [
        (sd, pt, gp, np_, nr)
        for sd in args.start_dates
        for (pt, gp) in variants
        for np_ in args.n_particles
        for nr in args.n_reps
    ]
    logger.info(
        f"[sweep] total runs: {len(runs)} "
        f"({len(args.start_dates)} dates x {len(variants)} priors x "
        f"{len(args.n_particles)} N x {len(args.n_reps)} reps)"
    )
    rows = [
        run_one(cfg, sd, pt, gp, np_, nr, output_root, args.predict, log_path)
        for (sd, pt, gp, np_, nr) in runs
    ]

    # Sort by best held-out test logZ per match (higher = better).
    rows_sorted = sorted(rows, key=lambda r: r["best_test_logz_per_match"], reverse=True)
    summary_df = pd.DataFrame(rows_sorted)
    summary_path = os.path.join(output_root, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    with open(os.path.join(output_root, "summary.json"), "w") as f:
        json.dump(rows_sorted, f, indent=2)

    logger.info("\n=== SWEEP SUMMARY (sorted by best held-out test logZ/match) ===")
    cols = ["training_start_date", "prior", "n_particles", "n_reps",
            "train_matches", "best_test_logz",
            "best_test_logz_per_match", "best_test_epoch"]
    if args.predict:
        cols += ["prediction_avg_loglik", "prediction_winner_accuracy"]
    for line in summary_df[cols].to_string(index=False).splitlines():
        logger.info(line)
    logger.info(
        f"Best config (test logZ/match): start={rows_sorted[0]['training_start_date']} "
        f"prior={rows_sorted[0]['prior']} "
        f"N={rows_sorted[0]['n_particles']} reps={rows_sorted[0]['n_reps']} "
        f"({rows_sorted[0]['best_test_logz_per_match']:.4f} logZ/match)"
    )
    logger.info(f"Saved summary to {os.path.abspath(summary_path)}")


if __name__ == "__main__":
    main()

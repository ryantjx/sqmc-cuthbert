"""Observe the filtered attack/defense states of specific teams over time.

Runs the RBPF forward filter over the FULL sequence (train + test + the World
Cup 2026 prediction window), extracts the weighted particle-mean of each team's
attack/defense state for every day, and plots a 2-row time series:

    row 1 : attack state over time
    row 2 : defense state over time

Each plotted team has its own line. Vertical lines mark the dates the team
played in the World Cup 2026 prediction window, labeled as
``"Team vs Opponent (team_score - opponent_score)"`` so you can see how a
result shifts the team's level.

The optimized parameters and the output directory are hard-coded to a specific
trained run (see ``PARAMS_PATH`` / ``OUT_DIR`` below). Point them at a newer
run if you retrain.

Usage:
    python rbsqmc/comparison/sqmc_smc/observe_team_states.py [--teams Spain Argentina]
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

import jax

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from rbsqmc.src.data.data import get_training_data, concat_football_results
from rbsqmc.src.model.rbsmc.optimization import run_filter_unbiased
from rbsqmc.src.utils.graphic import plot_all
from rbsqmc.src.utils.helpers import load_params, resolve_teams

# ---------------------------------------------------------------------------
# Hardcoded paths -- point these at the trained run you want to inspect.
# ---------------------------------------------------------------------------
# The directory produced by train_model.py, e.g. gd_<timestamp>/.
RUN_DIR = "rbsqmc/outputs/train_model/gd_20260823_094237"
PARAMS_PATH = f"{RUN_DIR}/best_params.json"
OUT_DIR = f"{RUN_DIR}/prediction"
CFG_PATH = f"{RUN_DIR}/run_config.json"
# Reuse the training config for consistency (plain dict, not EM params).
with open(CFG_PATH) as _f:
    CFG = json.load(_f)

# ---------------------------------------------------------------------------
# Config (mirrors train_model.py)
# ---------------------------------------------------------------------------
# CFG = {
#     "training_start_date": "1950-01-01",
#     "test_start_date": "2024-01-01",
#     "prediction_start_date": "2026-06-11",
#     "n_particles": 250,          # N
#     "max_goals": 8,              # MAX_GOALS
#     "seed": 0,                   # PRNG seed
#     "include_friendly": True,
#     "teams": "worldcup2026",
# }

DEFAULT_TEAMS = ["Spain", "Argentina"]


def parse_args(argv: list[str]) -> tuple[list[str], float]:
    """Parse ``--teams`` and ``--gamma-scale`` from argv.

    ``--gamma-scale <s>`` multiplies the prior state covariance ``gamma_0`` by
    ``s`` (> 1 inflates it). A larger covariance raises the Kalman gain, so a
    single match result moves the attack/defense strengths more visibly. This
    is a diagnostic knob; the default is ``1.0`` (use the fitted params as-is).
    """
    teams = list(DEFAULT_TEAMS)
    gamma_scale = 1.0
    argv = list(argv)
    if "--gamma-scale" in argv:
        i = argv.index("--gamma-scale")
        if i + 1 < len(argv):
            gamma_scale = float(argv[i + 1])
    if "--teams" in argv:
        i = argv.index("--teams")
        rest = argv[i + 1:]
        # consume consecutive non-flag arguments as team names
        names = []
        for arg in rest:
            if arg.startswith("--") or arg == "--gamma-scale":
                break
            names.append(arg)
        if names:
            teams = names
    return teams, gamma_scale


def weighted_filter_means(x, log_weights):
    """Weighted particle-mean of team states per time step.

    Args:
        x: ``(T, N, num_teams, 2)`` particle states.
        log_weights: ``(T, N)`` particle log-weights.

    Returns:
        ``(T, num_teams, 2)`` weighted mean (attack/defense in last axis).
    """
    finite_lw = np.where(np.isfinite(log_weights), log_weights, -np.inf)
    shifted = np.exp(log_weights - np.max(finite_lw, axis=1, keepdims=True))
    shifted[~np.isfinite(shifted)] = 0.0
    weight_sum = shifted.sum(axis=1, keepdims=True)
    uniform = np.full_like(shifted, 1.0 / x.shape[1])
    weights = np.divide(shifted, weight_sum, out=uniform, where=weight_sum > 0)
    return np.sum(x[..., :2] * weights[:, :, None, None], axis=1)


def collect_team_fixtures(prediction_inputs, team_id, team_id_to_name):
    """Return ``(date, label)`` for every WC2026 match the team played in.

    ``label`` is ``"Team vs Opponent (team_score - opp_score)"``.
    """
    fixtures = []
    home = np.asarray(prediction_inputs.matches.home_id)
    away = np.asarray(prediction_inputs.matches.away_id)
    home_score = np.asarray(prediction_inputs.matches.home_score)
    away_score = np.asarray(prediction_inputs.matches.away_score)
    mask = np.asarray(prediction_inputs.match_mask)
    date = np.asarray(prediction_inputs.date)

    team_name = team_id_to_name[int(team_id)]
    for t in range(home.shape[0]):
        for j in range(home.shape[1]):
            if not mask[t, j]:
                continue
            if home[t, j] == team_id:  # team is home
                opp = team_id_to_name[int(away[t, j])]
                ts, os_ = home_score[t, j], away_score[t, j]
            elif away[t, j] == team_id:  # team is away
                opp = team_id_to_name[int(home[t, j])]
                os_, ts = home_score[t, j], away_score[t, j]
            else:
                continue
            label = f"{team_name} vs {opp} ({ts} - {os_})"
            fixtures.append((date[t], label))
    return fixtures


def plot_team_states(
    dates_axis,
    means,
    team_ids,
    team_id_to_name,
    fixtures_by_team,
    save_path,
    draw_lines=False,
    xlim=None,
):
    """Plot 3 rows (attack / defense / total) for each team.

    Args:
        dates_axis: x-axis values (one per time step, including the prepended
            initial state).
        means: ``(T+1, n_teams, 2)`` weighted particle-mean states.
        fixtures_by_team: ``{team_id: [(date, label), ...]}`` for prediction-
            window fixtures (only used when ``draw_lines`` is True).
        draw_lines: if True, draw dashed vertical lines + score labels at each
            fixture date. When several teams play on the same day, the rotated
            labels are staggered vertically so they do not overlap.
        xlim: optional ``(xmin, xmax)`` to zoom the x-axis (e.g. to the
            prediction phase).
    """
    # Solid, clearly-distinct colours for each team (and their labels/lines).
    SOLID_COLORS = [
        "#D62728",  # red
        "#1F77B4",  # blue
        "#2CA02C",  # green
        "#9467BD",  # purple
        "#FF7F0E",  # orange
        "#8C564B",  # brown
        "#17BECF",  # cyan
        "#E377C2",  # pink
        "#7F7F7F",  # gray
        "#BCBD22",  # olive
    ]
    colors = SOLID_COLORS[: len(team_ids)]
    fig, (attack_ax, defense_ax, total_ax) = plt.subplots(
        3, 1, figsize=(14, 12), sharex=True
    )

    for team_id, color in zip(team_ids, colors):
        name = team_id_to_name[int(team_id)]
        attack_ax.plot(dates_axis, means[:, int(team_id), 0],
                       color=color, linewidth=2.2, label=name)
        defense_ax.plot(dates_axis, means[:, int(team_id), 1],
                        color=color, linewidth=2.2, label=name)
        total_ax.plot(
            dates_axis,
            means[:, int(team_id), 0] + means[:, int(team_id), 1],
            color=color, linewidth=2.2, label=name,
        )

    if draw_lines:
        # Flatten all (date, label, color) fixtures across teams and group by
        # day so that same-day labels can be staggered to avoid overlap.
        annotations = [
            (date, label, colors[team_ids.index(team_id)])
            for team_id in team_ids
            for date, label in fixtures_by_team[int(team_id)]
        ]
        by_day: dict[int, list] = {}
        for date, label, color in annotations:
            by_day.setdefault(int(date), []).append((date, label, color))

        for day_group in by_day.values():
            # Vertical (rotated) labels extend upward roughly proportional to
            # their text length, so stagger same-day labels by an amount large
            # enough to clear the tallest label rather than a fixed tiny step.
            max_len = max(len(label) for _, label, _ in day_group)
            step = 0.04 + max_len * 0.014  # axes-fraction per label
            for i, (date, label, color) in enumerate(day_group):
                y_frac = 0.02 + i * step
                for ax in (attack_ax, defense_ax, total_ax):
                    ax.axvline(date, color=color, linestyle="--", alpha=0.6)
                total_ax.annotate(
                    label,
                    xy=(date, y_frac),
                    xycoords=("data", "axes fraction"),
                    rotation=90,
                    fontsize=7,
                    color=color,
                    ha="right",
                    va="bottom",
                )

    total_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    total_ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

    if xlim is not None:
        for ax in (attack_ax, defense_ax, total_ax):
            ax.set_xlim(*xlim)

    attack_ax.set_ylabel("Attack state")
    defense_ax.set_ylabel("Defense state")
    total_ax.set_ylabel("Total strength (attack + defense)")
    total_ax.set_xlabel("Date")
    for ax in (attack_ax, defense_ax, total_ax):
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    attack_ax.set_title(
        "Filtered attack/defense/total states  ·  dashed lines = match in "
        "WC2026 prediction window" if draw_lines else "Filtered attack/defense/total states"
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {os.path.abspath(save_path)}")
    return fig, (attack_ax, defense_ax, total_ax)


def main():
    teams, gamma_scale = parse_args(sys.argv[1:])

    params = load_params(PARAMS_PATH)

    # Diagnostic knob: inflate gamma_0 (the prior state covariance) to raise
    # the Kalman gain, making individual match results move strengths more.
    if gamma_scale != 1.0:
        params = params._replace(
            gamma_0=params.gamma_0 * gamma_scale
        )
        print(f"Inflating gamma_0 by {gamma_scale:.3f}x for the diagnostic run")

    # Scale-specific filenames so a diagnostic run doesn't overwrite baseline.
    suffix = f"_gs{gamma_scale:g}" if gamma_scale != 1.0 else ""

    key = jax.random.PRNGKey(CFG["seed"])
    teams_only = resolve_teams(CFG)
    (_, _, _), (train_inputs, test_inputs, prediction_inputs), team_id_to_name = get_training_data(
        train_start_date=CFG["training_start_date"],
        test_start_date=CFG["test_start_date"],
        prediction_start_date=CFG["prediction_start_date"],
        max_goals=CFG["max_goals"],
        include_friendly=CFG["include_friendly"],
        teams_only=teams_only,
    )

    # Build the full sequence the filter runs over.
    observed_inputs = concat_football_results(train_inputs, test_inputs)
    full_inputs = concat_football_results(observed_inputs, prediction_inputs)

    # Number of observed (train+test) days == first prediction day index.
    n_obs_days = observed_inputs.timestamp.shape[0]

    print(f"Filtering {full_inputs.timestamp.shape[0]} days (train+test+prediction)...")
    filtered_states, model_inputs_rbpf = run_filter_unbiased(
        key=key,
        model_inputs=full_inputs,
        params=params,
        n_particles=CFG["n_particles"],
        max_goals=CFG["max_goals"],
    )

    x = np.asarray(filtered_states.particles.x)      # (T+1, N, n_teams, 2)
    lw = np.asarray(filtered_states.log_weights)     # (T+1, N)
    means = weighted_filter_means(x, lw)             # (T+1, n_teams, 2)

    # Align the date axis with the prepended initial state (index 0).
    dates = np.asarray(full_inputs.date)
    dates_axis = np.concatenate([dates[:1], dates])  # (T+1,)

    # Resolve team ids and collect fixtures.
    name_to_id = {name: i for i, name in team_id_to_name.items()}
    team_ids = [name_to_id[name] for name in teams if name in name_to_id]
    missing = [name for name in teams if name not in name_to_id]
    if missing:
        print(f"Warning: unknown team names (skipped): {missing}")

    fixtures_by_team = {
        int(tid): collect_team_fixtures(prediction_inputs, tid, team_id_to_name)
        for tid in team_ids
    }

    # ---- Export selected teams' state series for quantitative comparison ----
    export_path = os.path.join(OUT_DIR, f"team_states{suffix}.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(export_path, "w") as f:
        json.dump(
            {
                "teams": [team_id_to_name[int(tid)] for tid in team_ids],
                "dates": [str(np.datetime64(int(d), "D")) for d in dates_axis],
                "attack": {
                    team_id_to_name[int(tid)]: means[:, int(tid), 0].astype(float).tolist()
                    for tid in team_ids
                },
                "defense": {
                    team_id_to_name[int(tid)]: means[:, int(tid), 1].astype(float).tolist()
                    for tid in team_ids
                },
            },
            f,
            indent=2,
        )
    print(f"Saved team states to {os.path.abspath(export_path)}")

    # ---- Plot 1: full train + test + prediction range, no fixture lines ----
    full_path = os.path.join(OUT_DIR, f"observe_team_states_full{suffix}.png")
    plot_team_states(
        dates_axis=dates_axis,
        means=means,
        team_ids=team_ids,
        team_id_to_name=team_id_to_name,
        fixtures_by_team=fixtures_by_team,
        save_path=full_path,
        draw_lines=False,
    )

    # ---- Plot 2: prediction phase zoomed, with fixture lines --------------
    pred_start = dates[n_obs_days]          # first prediction day date
    pred_end = dates[-1]                    # last day in the full sequence
    pred_path = os.path.join(OUT_DIR, f"observe_team_states{suffix}.png")
    plot_team_states(
        dates_axis=dates_axis,
        means=means,
        team_ids=team_ids,
        team_id_to_name=team_id_to_name,
        fixtures_by_team=fixtures_by_team,
        save_path=pred_path,
        draw_lines=True,
        xlim=(pred_start, pred_end),
    )

    # ---- Plot 3: final states / rankings / top strengths AFTER prediction --
    # filtered_states already spans train+test+prediction, so its final step
    # is the post-prediction state. Reuse plot_all to reproduce the same
    # outputs as train_model's final_filter/ (top_strengths, final_rankings,
    # timeseries_states, correlation, filter_states.npz).
    post_dir = os.path.join(OUT_DIR, "final_filter_after_prediction")
    plot_all(
        filtered_states=filtered_states,
        augmented_results=model_inputs_rbpf,  # post-prediction augmented results
        team_id_to_name=team_id_to_name,
        top_n=10,
        save_path=post_dir,
        timestamps=dates_axis,
        params=params,
    )


if __name__ == "__main__":
    main()

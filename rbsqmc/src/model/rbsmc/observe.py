"""Post-prediction observe phase for the RBPF football model.

This module plots selected teams' filtered attack/defense/total states over the
full train+test+prediction sequence and saves the final filter-after-prediction
diagnostics (``final_filter_after_prediction/`` folder via ``plot_all``).

``run_observe`` is method-aware: pass ``method="smc"`` to filter with the
RB-SMC forward filter, or ``method="sqmc"`` to filter with the RB-SQMC forward
filter. Both paths share the same plotting helpers and produce identical output
artifacts.
"""

import json
import os
from typing import NamedTuple

import jax
import numpy as np

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from rbsqmc.src.data.data import concat_football_results
from rbsqmc.src.model.rbsqmc.model_rbsqmc import run_filter_sqmc
from rbsqmc.src.model.rbsmc.optimization import run_filter_unbiased
from rbsqmc.src.model.rbsmc.train_model_gpu import prepare_data
from rbsqmc.src.utils.graphic import plot_all
from rbsqmc.src.utils.type import RBPFState

# ---------------------------------------------------------------------------
# Adapter exposing SQMC filter outputs through the SMC state interface
# ---------------------------------------------------------------------------
class SQMCFilteredStates(NamedTuple):
    """Adapter exposing SQMC arrays through the interface used by plot_all."""

    particles: RBPFState
    log_weights: jax.Array
    log_normalizing_constant: jax.Array


# ---------------------------------------------------------------------------
# observe phase: plot selected teams' filtered attack/defense/total states
# ---------------------------------------------------------------------------
OBSERVE_DEFAULT_TEAMS = ["Spain", "England", "France", "Argentina"]

# Solid, clearly-distinct colours for each team (and their labels/lines).
OBSERVE_COLORS = [
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


def _observe_weighted_filter_means(x, log_weights):
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


def _observe_collect_team_fixtures(prediction_inputs, team_id, team_id_to_name):
    """Return ``[(date, label), ...]`` for every prediction-window match.

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


def _observe_plot_team_states(
    dates_axis,
    means,
    team_ids,
    team_id_to_name,
    fixtures_by_team,
    save_path,
    draw_lines=False,
    xlim=None,
):
    """Plot 3 rows (attack / defense / total) for each team."""
    colors = OBSERVE_COLORS[: len(team_ids)]
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
        # Flatten fixtures across teams and group by day so same-day labels
        # can be staggered vertically to avoid overlap.
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
        "prediction window" if draw_lines else "Filtered attack/defense/total states"
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {os.path.abspath(save_path)}")


def _run_forward_filter(method: str, key, full_inputs, cfg, params):
    """Run the chosen forward filter over the full input sequence.

    Returns ``(filtered_states, model_inputs_rbpf)`` where ``filtered_states``
    exposes ``.particles.x`` and ``.log_weights`` for both methods.
    """
    if method == "smc":
        filtered_states, model_inputs_rbpf = run_filter_unbiased(
            key=key,
            model_inputs=full_inputs,
            params=params,
            n_particles=cfg["n_particles"],
            max_goals=cfg["max_goals"],
        )
    elif method == "sqmc":
        result, model_inputs_rbpf = run_filter_sqmc(
            key=key,
            model_inputs=full_inputs,
            params=params,
            n_particles=cfg["n_particles"],
            max_goals=cfg["max_goals"],
        )
        filtered_states = SQMCFilteredStates(
            particles=RBPFState(x=result["particles_x"]),
            log_weights=result["log_weights"],
            log_normalizing_constant=result["log_normalizing_constant"],
        )
    else:
        raise ValueError(f"Unknown observe method: {method}")
    return filtered_states, model_inputs_rbpf


def run_observe(
    cfg: dict,
    params,
    output_dir: str,
    method: str = "smc",
) -> None:
    """Local phase: plot selected teams' states over the full sequence.

    Produces three sets of outputs in ``output_dir``:
      * ``observe_team_states_full.png``   full train+test+prediction, no lines
      * ``observe_team_states.png``        prediction phase zoomed + fixture lines
      * ``observe_team_states.json``       per-team state series (quantitative)
    and a ``final_filter_after_prediction/`` folder via ``plot_all`` (final
    rankings, top strengths, correlation, filter states).

    Args:
        cfg: run configuration (must contain the data-split dates, ``seed``,
            ``n_particles``, and ``max_goals``).
        params: fitted ``EMParams``.
        output_dir: directory to write the artifacts into.
        method: ``"smc"`` uses the RB-SMC forward filter, ``"sqmc"`` uses the
            RB-SQMC forward filter.
    """
    teams = list(cfg.get("observe_teams") or OBSERVE_DEFAULT_TEAMS)
    gamma_scale = float(cfg.get("observe_gamma_scale", 1.0))

    # Diagnostic knob: inflate gamma_0 (prior state covariance) to raise the
    # Kalman gain, making individual match results move strengths more.
    if gamma_scale != 1.0:
        params = params._replace(gamma_0=params.gamma_0 * gamma_scale)
        print(f"Observe: inflating gamma_0 by {gamma_scale:.3f}x")
    suffix = f"_gs{gamma_scale:g}" if gamma_scale != 1.0 else ""

    (
        _train_df, _test_df, _pred_df,
        train_model_inputs, test_model_inputs, prediction_model_inputs,
        team_id_to_name, _init_params,
    ) = prepare_data(cfg)

    observed_inputs = concat_football_results(train_model_inputs, test_model_inputs)
    full_inputs = concat_football_results(observed_inputs, prediction_model_inputs)
    n_obs_days = observed_inputs.timestamp.shape[0]

    key = jax.random.PRNGKey(cfg["seed"])
    print(f"Observe[{method}]: filtering {full_inputs.timestamp.shape[0]} days "
          "(train+test+prediction)...")
    filtered_states, model_inputs_rbpf = _run_forward_filter(
        method=method,
        key=key,
        full_inputs=full_inputs,
        cfg=cfg,
        params=params,
    )

    x = np.asarray(filtered_states.particles.x)      # (T+1, N, n_teams, 2)
    lw = np.asarray(filtered_states.log_weights)     # (T+1, N)
    means = _observe_weighted_filter_means(x, lw)    # (T+1, n_teams, 2)

    dates = np.asarray(full_inputs.date)
    dates_axis = np.concatenate([dates[:1], dates])  # align with initial state

    # Resolve requested team ids.
    name_to_id = {name: i for i, name in team_id_to_name.items()}
    team_ids = [name_to_id[name] for name in teams if name in name_to_id]
    missing = [name for name in teams if name not in name_to_id]
    if missing:
        print(f"Observe[{method}]: warning, unknown team names (skipped): {missing}")
    if not team_ids:
        raise ValueError("No requested teams found in team_id_to_name")

    fixtures_by_team = {
        int(tid): _observe_collect_team_fixtures(
            prediction_model_inputs, tid, team_id_to_name
        )
        for tid in team_ids
    }

    # Quantitative export of the selected teams' state series.
    export_path = os.path.join(output_dir, f"observe_team_states{suffix}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(export_path, "w") as f:
        json.dump({
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
        }, f, indent=2)
    print(f"Observe[{method}]: saved team states to {os.path.abspath(export_path)}")

    # Plot 1: full train + test + prediction range, no fixture lines.
    _observe_plot_team_states(
        dates_axis, means, team_ids, team_id_to_name, fixtures_by_team,
        os.path.join(output_dir, f"observe_team_states_full{suffix}.png"),
        draw_lines=False,
    )
    # Plot 2: prediction phase zoomed, with fixture lines.
    _observe_plot_team_states(
        dates_axis, means, team_ids, team_id_to_name, fixtures_by_team,
        os.path.join(output_dir, f"observe_team_states{suffix}.png"),
        draw_lines=True,
        xlim=(dates[n_obs_days], dates[-1]),
    )
    # Plot 3: final states / rankings / top strengths AFTER prediction.
    plot_all(
        filtered_states=filtered_states,
        augmented_results=model_inputs_rbpf,
        team_id_to_name=team_id_to_name,
        top_n=10,
        save_path=os.path.join(output_dir, "final_filter_after_prediction"),
        timestamps=dates_axis,
        params=params,
    )

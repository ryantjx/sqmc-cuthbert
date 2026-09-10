"""Plots for the SQMC–EKF comparison.

Reuses the shared graphics utilities where a particle ``FilterStates``-like
object applies, and draws EKF-native equivalents for the correlated-SQMC-only
diagnostics (initial/final correlation matrices). The EKF has no particles, so
we wrap its per-team Gaussian means as a single-particle ``FilterStates`` to
reuse the ranking / strength / trajectory routines.
"""

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np

from rbsqmc.src.utils import graphic as g
from rbsqmc.src.utils.type import RBPFState


def _relocate_json(output_dir):
    """Move any ``.json`` data files written beside PNGs into ``output_dir/json/``.

    The shared graphics utilities persist per-plot data as ``<name>.json`` next
    to the ``<name>.png``. We keep PNGs in ``images/`` and move the JSON into a
    dedicated ``images/json/`` subfolder so the two artifact types stay separate.
    """
    json_dir = os.path.join(output_dir, "json")
    for name in os.listdir(output_dir):
        if name.endswith(".json"):
            os.makedirs(json_dir, exist_ok=True)
            shutil.move(os.path.join(output_dir, name), os.path.join(json_dir, name))


class MeanFilterStates:
    """Minimal FilterStates-compatible wrapper around EKF Gaussian means.

    Presenting the per-team marginal mean as a single deterministic particle
    lets the ranking / strength / trajectory functions run unchanged; the
    covariance is recorded separately for reference.
    """

    def __init__(self, mean, cov=None):
        # mean: (T+1, num_teams, 2), cov: (T+1, num_teams, 2, 2)
        self.particles = RBPFState(x=np.asarray(mean)[:, None, :, :])
        self.log_weights = np.zeros((np.asarray(mean).shape[0], 1))
        self.log_normalizing_constant = None
        self.ekf_cov = None if cov is None else np.asarray(cov)
        self.ekf_mean = np.asarray(mean)


class SQMCPosterior:
    """Weighted SQMC posterior moments, including residual Gaussian covariance."""


def weighted_sqmc_moments(result):
    """Mean and (residual) covariance of the SQMC particle posterior.

    The particles carry per-team marginal Gaussian *mean* coordinates after
    Rao--Blackwellization; the covariance is recovered from the shared B and
    the per-team gamma (the RB covariance contribution). We report the
    particle-weighted first moment as the posterior mean.
    """
    particles_x = np.asarray(result["particles_x"])          # (T+1, N, M, 2)
    log_weights = np.asarray(result["log_weights"])          # (T+1, N)
    n_steps, n_particles, n_teams, _ = particles_x.shape
    means = []
    for t in range(n_steps):
        w = np.exp(log_weights[t])
        w = w / w.sum()
        means.append(np.sum(particles_x[t] * w[:, None, None], axis=0))
    return np.array(means)  # (T+1, M, 2)


def wrap_sqmc(result, num_teams):
    """Expose SQMC results through the MeanFilterStates-like interface."""
    mean = weighted_sqmc_moments(result)
    return MeanFilterStates(mean)


def plot_convergence(ekf_history, sqmc_history, output_dir):
    """Train/test logZ overlay for both methods (per-epoch progress)."""
    os.makedirs(output_dir, exist_ok=True)
    plot_logz_overlay(ekf_history, sqmc_history, output_dir)


def plot_logz_overlay(ekf_history, sqmc_history, output_dir):
    """Two-panel EKF-vs-SQMC logZ overlay: train (left) and test (right) per epoch."""
    os.makedirs(output_dir, exist_ok=True)
    epoch = [r["epoch"] for r in ekf_history]
    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(14, 6))

    ax_train.plot(epoch, [r["train_logz"] for r in ekf_history],
                  label="EKF train logZ", color="steelblue", marker="o")
    ax_train.plot(epoch, [r["train_logz"] for r in sqmc_history],
                  label="SQMC train logZ", color="darkred", marker="s", linestyle="--")
    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Train Log Marginal Likelihood (logZ)")
    ax_train.set_title("EKF vs SQMC Train LogZ per Epoch")
    ax_train.legend()
    ax_train.grid(True, alpha=0.3)

    ax_test.plot(epoch, [r["test_logz"] for r in ekf_history],
                 label="EKF test logZ", color="steelblue", marker="o")
    ax_test.plot(epoch, [r["test_logz"] for r in sqmc_history],
                 label="SQMC test logZ", color="darkred", marker="s", linestyle="--")
    ax_test.set_xlabel("Epoch")
    ax_test.set_ylabel("Test Log Marginal Likelihood (logZ)")
    ax_test.set_title("EKF vs SQMC Test LogZ per Epoch")
    ax_test.legend()
    ax_test.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "logz_overlay_train_test.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_prediction(records, max_goals, output_dir, last_n=7):
    """Prediction heatmaps for the final ``last_n`` matches only.

    The prediction split is chronological, so the last 7 records are the World
    Cup knockout stage (quarters, semis, and final). Earlier group-stage
    heatmaps are omitted to keep the artifact set focused.
    """
    os.makedirs(output_dir, exist_ok=True)
    tail = records[-last_n:] if last_n else records
    g.plot_all_predictions(
        {"predictions": tail},
        max_goals=max_goals,
        save_path=os.path.join(output_dir, "prediction_plots"),
    )


def plot_correlation(sqmc_params, sqmc_augmented, team_id_to_name, output_dir):
    """SQMC correlation diagnostics.

    Only the top-N correlation bar chart is kept (used in the dissertation
    appendix); the full matrix heatmaps are omitted.
    """
    os.makedirs(output_dir, exist_ok=True)
    g.plot_correlation_topn_bar(
        sqmc_augmented, team_id_to_name,
        save_path=os.path.join(output_dir, "sqmc_correlation_topn_bar.png"),
    )
    _relocate_json(output_dir)


def plot_ranking_trajectory(states, team_id_to_name, timestamps, output_dir,
                            name="states", worldcup_team_ids=None,
                            pre_index=None, post_index=None):
    """Rankings, top strengths, and time-series trajectories for one method.

    The dataset is already restricted to the 2026 World Cup teams (config
    ``teams: worldcup2026``), so ``team_id_to_name`` contains only WC teams and
    every ranking here is a World Cup ranking. ``pre_index`` / ``post_index``
    are state indices (into ``states.particles.x``) at which to draw the pre-
    and post-World Cup rankings. ``states.particles.x`` has shape
    ``(T+1, 1, M, 2)`` with index ``i`` the state *before* match ``i``
    (index 0 = prior), so ``pre_index`` is the first prediction-split match and
    ``post_index`` is the final state.
    """
    os.makedirs(output_dir, exist_ok=True)
    g.plot_top_strengths(
        states, team_id_to_name, top_n=5,
        save_path=os.path.join(output_dir, f"{name}_top5_strengths.png"),
    )
    g.plot_timeseries_states(
        states, team_id_to_name, top_n=5, rank_by="total",
        timestamps=timestamps,
        save_path=os.path.join(output_dir, f"{name}_timeseries_states.png"),
    )
    plot_rankings_at_index(states, team_id_to_name, output_dir, name=name,
                           index=pre_index, label="pre_worldcup")
    plot_rankings_at_index(states, team_id_to_name, output_dir, name=name,
                           index=post_index, label="post_worldcup")
    _relocate_json(output_dir)


def plot_rankings_at_index(states, team_id_to_name, output_dir, name="states",
                           index=None, label="rankings"):
    """Rank all teams by total strength at a specific state index.

    ``index`` is an index into ``states.particles.x`` (shape ``(T+1, 1, M, 2)``).
    If ``index`` is None, the final state (``-1``) is used. The dataset is
    already World Cup-only, so this is a World Cup ranking.
    """
    os.makedirs(output_dir, exist_ok=True)
    if index is None:
        index = -1
    # Build a shallow wrapper that reports the state at ``index`` as its final
    # state, so the shared plot_final_rankings ranks that time slice.
    class _Sliced:
        def __init__(self, particles):
            self.particles = particles
    sliced = _Sliced(RBPFState(x=np.asarray(states.particles.x)[index][None]))
    g.plot_final_rankings(
        sliced, team_id_to_name,
        save_path=os.path.join(output_dir, f"{name}_{label}_rankings.png"),
    )
    _relocate_json(output_dir)

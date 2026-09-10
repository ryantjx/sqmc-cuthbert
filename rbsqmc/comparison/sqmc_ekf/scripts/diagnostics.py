"""Per-forecast SQMC diagnostics for reproducibility and explanation.

Given the single shared evaluation filter history (``result`` from
``run_filter_sqmc``) and the per-match scales, this module writes
``sqmc_prediction_diagnostics.json`` with one entry per forecast and a compact
NPZ of the final fixture's predictive coordinates and posterior weights.

The diagnostics distinguish predictive (uniform pre-likelihood weights) from
posterior (likelihood-weighted) summaries, report the particlewise
total-strength difference distribution, ESS before/after the current update,
and the raw (pre-normalization) score-grid mass. This is what makes the
Spain--Argentina reversal reproducible and explainable rather than assumed.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from rbsqmc.src.model.rbsmc.predict import predict_match_score_with_mass


def _ess(log_weights):
    """Effective sample size from normalized weights: 1 / sum(w**2)."""
    w = jnp.exp(log_weights - logsumexp(log_weights))
    return float(1.0 / jnp.sum(w**2))


def _outcome_probs(grid):
    """Home/draw/away probabilities from a normalized (G, G) score grid."""
    G = grid.shape[0]
    ii, jj = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    return {
        "home": float(grid[ii > jj].sum()),
        "draw": float(grid[ii == jj].sum()),
        "away": float(grid[ii < jj].sum()),
    }


def _team_means(particles_x, log_weights, home_id, away_id):
    """Weighted attack/defence means for the two playing teams.

    ``particles_x`` is ``(N, num_teams, 2)``; ``log_weights`` is ``(N,)``.
    Returns a dict with home/away attack and defence means.
    """
    w = jnp.exp(log_weights - logsumexp(log_weights))
    home = jnp.sum(particles_x[:, home_id, :] * w[:, None], axis=0)
    away = jnp.sum(particles_x[:, away_id, :] * w[:, None], axis=0)
    return {
        "home_attack": float(home[0]), "home_defence": float(home[1]),
        "away_attack": float(away[0]), "away_defence": float(away[1]),
    }


def _total_diff_distribution(particles_x, home_id, away_id):
    """Particlewise ``(att_home+def_home) - (att_away+def_away)`` summary."""
    diff = (
        particles_x[:, home_id, 0] + particles_x[:, home_id, 1]
        - particles_x[:, away_id, 0] - particles_x[:, away_id, 1]
    )
    diff = np.asarray(diff)
    return {
        "mean": float(diff.mean()),
        "q05": float(np.quantile(diff, 0.05)),
        "q50": float(np.quantile(diff, 0.50)),
        "q95": float(np.quantile(diff, 0.95)),
        "fraction_positive": float((diff > 0).mean()),
    }


def build_diagnostics(
    dataset, params, cfg, result, scales, source_revision, dataset_hash,
    filter_key_provenance, schema_version, scope="worldcup", checkpoint_hash=None,
):
    """Build the per-forecast diagnostics list and the final-fixture NPZ data.

    ``result`` is the shared evaluation filter dict (``particles_x``,
    ``log_weights``). ``scales`` is the ``(T, 1)`` per-match scale array used
    for that filter. Rows align with ``dataset.frame`` in the prediction-split
    range.

    ``scope`` controls which forecasts are persisted to the JSON diagnostics
    (the final-fixture NPZ is always written):
      - ``"all"``: every prediction-split match (can be large, ~5k rows).
      - ``"worldcup"`` (default): only 2026 World Cup fixtures.
      - ``"final"``: only the last fixture in the prediction split.

    ``checkpoint_hash`` (optional) is the SHA-256 of the fitted-params
    checkpoint that produced this filter; it binds the diagnostics to the exact
    parameters used.
    """
    rng_start = dataset.train_count + dataset.test_count
    frame_rows = dataset.frame.iloc[rng_start:]
    particles_x = result["particles_x"]
    log_weights = result["log_weights"]
    max_goals = cfg["max_goals"]
    n_particles = cfg["n_particles"]

    def _in_scope(p, row):
        if scope == "all":
            return True
        if scope == "final":
            return p == len(frame_rows) - 1
        # worldcup (default): 2026 FIFA World Cup fixtures.
        return bool(
            (row.get("tournament", "") == "FIFA World Cup")
            and (getattr(row.get("date"), "year", 0) == 2026)
        )

    entries = []
    final_npz = None
    for p, (_, row) in enumerate(frame_rows.iterrows()):
        t = rng_start + p
        home_id = int(row["home_id"])
        away_id = int(row["away_id"])
        scale = float(scales[t, 0])

        # Predictive: uniform pre-likelihood weights at history index t+1.
        x_pred = particles_x[t + 1]
        w_pred = jnp.zeros_like(log_weights[t + 1])
        grid, raw_mass = predict_match_score_with_mass(
            x_pred, w_pred, home_id, away_id,
            params["model"].alpha, params["model"].beta, max_goals, scale,
        )
        grid = np.asarray(grid)

        # Persist the final fixture's predictive coordinates + posterior weights.
        if p == len(frame_rows) - 1:
            final_npz = {
                "particles_x_predictive": np.asarray(x_pred),
                "log_weights_predictive": np.asarray(w_pred),
                "log_weights_posterior": np.asarray(log_weights[t + 1]),
                "home_id": np.asarray(home_id),
                "away_id": np.asarray(away_id),
                "scale": np.asarray(scale),
                "alpha": np.asarray(params["model"].alpha),
                "beta": np.asarray(params["model"].beta),
                "max_goals": np.asarray(max_goals),
            }

        if not _in_scope(p, row):
            continue

        # Posterior: likelihood-weighted means after the current update.
        post_means = _team_means(particles_x[t + 1], log_weights[t + 1], home_id, away_id)

        entry = {
            "fixture_id": int(row.get("source_row", p)),
            "date": row["date"].strftime("%Y-%m-%d"),
            "home_id": home_id,
            "away_id": away_id,
            "home": row["home_team"],
            "away": row["away_team"],
            "tournament": row.get("tournament", ""),
            "full_sequence_match_index": t,
            "history_index": t + 1,
            "current_scale": scale,
            "label": "predictive",
            "predictive_means": _team_means(x_pred, w_pred, home_id, away_id),
            "posterior_means": post_means,
            "total_strength_difference": _total_diff_distribution(
                x_pred, home_id, away_id
            ),
            "ess_before_resampling": _ess(log_weights[t]),
            "ess_after_update": _ess(log_weights[t + 1]),
            "raw_grid_mass": float(raw_mass),
            "outcome_probabilities": _outcome_probs(grid),
            "n_particles": n_particles,
            "max_goals": max_goals,
        }
        entries.append(entry)

    payload = {
        "schema_version": schema_version,
        "source_revision": source_revision,
        "dataset_hash": dataset_hash,
        "checkpoint_hash": checkpoint_hash,
        "filter_key_provenance": filter_key_provenance,
        "n_particles": n_particles,
        "max_goals": max_goals,
        "scope": scope,
        "forecasts": entries,
    }
    return payload, final_npz


def write_diagnostics(results_dir, payload, final_npz):
    """Atomically write the diagnostics JSON and the final-fixture NPZ."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    tmp = results_dir / "sqmc_prediction_diagnostics.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    tmp.replace(results_dir / "sqmc_prediction_diagnostics.json")
    np.savez(results_dir / "sqmc_final_fixture.npz", **final_npz)

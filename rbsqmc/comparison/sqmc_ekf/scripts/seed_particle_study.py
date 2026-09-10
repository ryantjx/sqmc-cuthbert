"""F8: fixed-parameter seed/particle-count sensitivity study.

Loads the fitted checkpoints from a completed comparison run, holds the SQMC
parameters fixed (no retraining), and re-runs the evaluation filter across a
recorded set of seeds and particle counts. For every evaluation it extracts the
Spain--Argentina forecast from the same filter history: predictive and
post-final means, the strength-difference distribution, outcome probabilities,
raw grid mass, and ESS. The output summarises the spread across seeds/counts so
any remaining mean-strength/win-probability disagreement can be explained by
the diagnostics rather than assumed.

Usage:
    python -m rbsqmc.comparison.sqmc_ekf.scripts.seed_particle_study \
        --run-dir <repaired run dir> \
        --seeds 0 1 2 \
        --particle-counts 128 512 2048 \
        [--output seed_particle_study.json]
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rbsqmc.comparison.sqmc_ekf.scripts import train as train_mod
from rbsqmc.comparison.sqmc_ekf.scripts.scaling import sqmc_match_scales
from rbsqmc.src.data.data_ekf import load_dataset as data_mod_load_dataset
from rbsqmc.src.model.rbsqmc.model_rbsqmc import run_filter_sqmc
from rbsqmc.src.model.rbsmc.predict import predict_match_score_with_mass
from jax.scipy.special import logsumexp


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
    """Weighted attack/defence means for the two playing teams."""
    w = jnp.exp(log_weights - logsumexp(log_weights))
    home = jnp.sum(particles_x[:, home_id, :] * w[:, None], axis=0)
    away = jnp.sum(particles_x[:, away_id, :] * w[:, None], axis=0)
    return {
        "home_attack": float(home[0]), "home_defence": float(home[1]),
        "away_attack": float(away[0]), "away_defence": float(away[1]),
    }


def _total_diff_distribution(particles_x, home_id, away_id, log_weights=None):
    """Summarize total-strength differences under predictive/posterior weights.

    Predictive quantiles retain NumPy's linear interpolation convention.
    Posterior quantiles use the inverse weighted empirical CDF: the smallest
    value whose cumulative normalized weight reaches the requested probability.
    Zero-weight particles are excluded from that CDF.
    """
    diff = (
        particles_x[:, home_id, 0] + particles_x[:, home_id, 1]
        - particles_x[:, away_id, 0] - particles_x[:, away_id, 1]
    )
    diff = np.asarray(diff)
    if log_weights is not None:
        lw = np.asarray(log_weights, dtype=float)
        if (lw.shape != diff.shape or np.isnan(lw).any()
                or np.isposinf(lw).any() or not np.isfinite(lw).any()):
            raise ValueError("Posterior log weights must match particles and have finite mass")
        weights = np.exp(lw - np.max(lw))
        weights /= weights.sum()
        order = np.argsort(diff, kind="stable")
        order = order[weights[order] > 0]
        cdf = np.cumsum(weights[order])
        cdf[-1] = 1.0
        quantiles = diff[order[np.searchsorted(cdf, [0.05, 0.5, 0.95], side="left")]]
        return {
            "mean": float(np.dot(weights, diff)),
            "q05": float(quantiles[0]),
            "q50": float(quantiles[1]),
            "q95": float(quantiles[2]),
            "fraction_positive": float(weights[diff > 0].sum()),
        }
    return {
        "mean": float(diff.mean()),
        "q05": float(np.quantile(diff, 0.05)),
        "q50": float(np.quantile(diff, 0.50)),
        "q95": float(np.quantile(diff, 0.95)),
        "fraction_positive": float((diff > 0).mean()),
    }


def _find_fixture(dataset, home_name, away_name):
    """Locate the target fixture in the prediction split; return (p, t, row)."""
    rng_start = dataset.train_count + dataset.test_count
    frame_rows = dataset.frame.iloc[rng_start:]
    for p, (_, row) in enumerate(frame_rows.iterrows()):
        if row["home_team"] == home_name and row["away_team"] == away_name:
            return p, rng_start + p, row
    raise ValueError(f"Fixture {home_name} vs {away_name} not found in the prediction split")


def evaluate_fixture(dataset, params, cfg, key, n_particles, p, t, scale):
    """Run one evaluation filter and extract the target fixture's diagnostics.

    The fitted parameters are held fixed; only the key (seed) and particle
    count vary. Predictive summaries use uniform pre-likelihood weights at
    history index t+1; post-final summaries use the updated weights.
    """
    scales = sqmc_match_scales(
        dataset.inputs.friendly, params["friendly_scale"], cfg.get("match_scale", 1.0),
    )
    result, _ = run_filter_sqmc(
        key, dataset.sqmc, params["model"], n_particles, cfg["max_goals"],
        match_scales=scales,
    )
    particles_x = result["particles_x"]
    log_weights = result["log_weights"]
    home_id = int(dataset.frame.iloc[t]["home_id"])
    away_id = int(dataset.frame.iloc[t]["away_id"])

    # Predictive: uniform pre-likelihood weights at history index t+1.
    x_pred = particles_x[t + 1]
    w_pred = jnp.zeros_like(log_weights[t + 1])
    grid, raw_mass = predict_match_score_with_mass(
        x_pred, w_pred, home_id, away_id,
        params["model"].alpha, params["model"].beta, cfg["max_goals"], scale,
    )
    grid = np.asarray(grid)

    return {
        "seed": int(key[1]),
        "n_particles": n_particles,
        "predictive_means": _team_means(x_pred, w_pred, home_id, away_id),
        "posterior_means": _team_means(particles_x[t + 1], log_weights[t + 1], home_id, away_id),
        "total_strength_difference": _total_diff_distribution(x_pred, home_id, away_id),
        "post_total_strength_difference": _total_diff_distribution(
            particles_x[t + 1], home_id, away_id, log_weights[t + 1]
        ),
        "ess_before_resampling": _ess(log_weights[t]),
        "ess_after_update": _ess(log_weights[t + 1]),
        "raw_grid_mass": float(raw_mass),
        "outcome_probabilities": _outcome_probs(grid),
    }


def summarize(entries):
    """Summarize the spread of key quantities across seeds and particle counts."""
    def _spread(key_path):
        values = []
        for e in entries:
            v = e
            for k in key_path:
                v = v[k]
            values.append(float(v))
        return {
            "min": min(values), "max": max(values),
            "mean": sum(values) / len(values),
        }

    summary = {
        "n_evaluations": len(entries),
        "predictive_total_difference": _spread(("total_strength_difference", "mean")),
        "predictive_home_win": _spread(("outcome_probabilities", "home")),
        "predictive_draw": _spread(("outcome_probabilities", "draw")),
        "predictive_away_win": _spread(("outcome_probabilities", "away")),
        "posterior_total_difference": _spread(("post_total_strength_difference", "mean")),
        "ess_after_update": _spread(("ess_after_update",)),
        "raw_grid_mass": _spread(("raw_grid_mass",)),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="F8 seed/particle sensitivity study")
    parser.add_argument("--run-dir", required=True, help="Repaired run directory with fitted checkpoints")
    parser.add_argument("--data", default="rbsqmc/data/results.csv", type=str)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--particle-counts", nargs="+", type=int, default=[128, 512, 2048])
    parser.add_argument("--home", default="Spain", type=str)
    parser.add_argument("--away", default="Argentina", type=str)
    parser.add_argument("--output", default=None, type=str)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    results_dir = run_dir / "results"
    cfg = json.loads((results_dir / "comparison_config.json").read_text())
    dataset = data_mod_load_dataset(args.data, cfg, smoke=bool(cfg.get("smoke")))

    # Load the fitted SQMC checkpoint; parameters are held fixed throughout.
    checkpoint_path = results_dir / "sqmc" / "fitted_params.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    stored_metadata = json.loads((results_dir / "dataset_metadata.json").read_text())
    if dataset.metadata["source_sha256"] != stored_metadata["source_sha256"]:
        raise ValueError("Study dataset does not match the fitted run's dataset hash")
    if checkpoint["team_id_to_name"] != {str(k): v for k, v in dataset.teams.items()}:
        raise ValueError("Study team mapping does not match the fitted checkpoint")
    params = train_mod.load_fitted_params(checkpoint_path, "sqmc")
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    p, t, row = _find_fixture(dataset, args.home, args.away)
    scales = sqmc_match_scales(
        dataset.inputs.friendly, params["friendly_scale"], cfg.get("match_scale", 1.0),
    )
    scale = float(scales[t, 0])

    entries = []
    for n_particles in args.particle_counts:
        for seed in args.seeds:
            key = jax.random.PRNGKey(seed)
            print(f"evaluating seed={seed}, particles={n_particles} ...", flush=True)
            entry = evaluate_fixture(dataset, params, cfg, key, n_particles, p, t, scale)
            entry["seed"] = seed
            entries.append(entry)
            probs = entry["outcome_probabilities"]
            print(
                f"  home={probs['home']:.4f} draw={probs['draw']:.4f} "
                f"away={probs['away']:.4f} ESS={entry['ess_after_update']:.1f} "
                f"mass={entry['raw_grid_mass']:.4f}",
                flush=True,
            )

    by_count = {}
    for n_particles in args.particle_counts:
        subset = [e for e in entries if e["n_particles"] == n_particles]
        by_count[str(n_particles)] = summarize(subset)

    payload = {
        "fixture": {"home": args.home, "away": args.away,
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "full_sequence_match_index": t, "prediction_index": p},
        "checkpoint_hash": checkpoint_hash,
        "dataset_hash": dataset.metadata.get("source_sha256"),
        "schema_version": 2,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[4], text=True
        ).strip(),
        "study_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "quantile_conventions": {
            "predictive": "numpy linear interpolation, uniform particle weights",
            "posterior": "inverse weighted empirical CDF, normalized posterior weights",
        },
        "n_particles_list": args.particle_counts,
        "seeds": args.seeds,
        "evaluations": entries,
        "summary_by_particle_count": by_count,
        "summary_all": summarize(entries),
    }
    output = Path(args.output) if args.output else results_dir / "seed_particle_study.json"
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    tmp.replace(output)
    print(f"study written to {output}", flush=True)


if __name__ == "__main__":
    main()
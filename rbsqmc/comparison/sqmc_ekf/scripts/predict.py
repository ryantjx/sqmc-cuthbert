"""Unified one-step-ahead prediction for the SQMC–EKF comparison.

Both methods forecast each prediction-split match *before* assimilating its
known result, then update.

* SQMC reuses ``run_sequential_predict_rbsqmc``, which filters the observed
  prefix once and, at each prediction step, forecasts from pre-likelihood
  particles (so the current score cannot enter its own forecast).
* EKF propagates the marginal of the two playing teams from the state *before*
  match ``i`` and integrates the bivariate-Poisson likelihood under the
  predictive Gaussian by Gauss--Hermite quadrature.

The returned grids/log-probabilities align one-to-one with the prediction-split
rows of ``dataset.frame``.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from rbsqmc.src.model.ekf import model as ekf
from rbsqmc.comparison.sqmc_ekf.scripts.scaling import sqmc_match_scales
from rbsqmc.src.model.rbsqmc.predict_rbsqmc import (
    predict_from_sqmc_history,
    run_sequential_predict_rbsqmc,
)
from rbsqmc.src.utils.type import FootballResults, Matches


def pred_index_range(dataset):
    """Held-out prediction rows: after train + test, to the end."""
    start = dataset.train_count + dataset.test_count
    return slice(start, dataset.inputs.score.shape[0])


def _slice_sqmc(sqmc, start, end):
    """Slice a match-level ``FootballResults`` to time rows ``[start, end)``."""
    return FootballResults(
        date=sqmc.date[start:end],
        timestamp=sqmc.timestamp[start:end],
        timestamp_prev=sqmc.timestamp_prev[start:end],
        matches=Matches(
            home_id=sqmc.matches.home_id[start:end, 0:1],
            away_id=sqmc.matches.away_id[start:end, 0:1],
            home_score=sqmc.matches.home_score[start:end, 0:1],
            away_score=sqmc.matches.away_score[start:end, 0:1],
        ),
        match_mask=jnp.ones((end - start, 1), dtype=bool),
    )


@dataclass
class MethodPredictions:
    grids: np.ndarray   # (P, G, G) normalized score grids for the prediction split
    logp: np.ndarray    # (P,) predictive log-probability of the actual score


def predict_sqmc(dataset, params, cfg, key):
    """One-step-ahead SQMC score grids over the prediction split."""
    rng = pred_index_range(dataset)
    observed = _slice_sqmc(dataset.sqmc, 0, rng.start)
    prediction = _slice_sqmc(dataset.sqmc, rng.start, dataset.sqmc.timestamp.shape[0])
    observed_scales = sqmc_match_scales(
        dataset.inputs.friendly[:rng.start], params["friendly_scale"],
        cfg.get("match_scale", 1.0),
    )
    prediction_scales = sqmc_match_scales(
        dataset.inputs.friendly[rng.start:], params["friendly_scale"],
        cfg.get("match_scale", 1.0),
    )

    grids, logp, _ = run_sequential_predict_rbsqmc(
        key=key,
        observed_inputs=observed,
        prediction_inputs=prediction,
        params=params["model"],
        n_particles=cfg["n_particles"],
        max_goals=cfg["max_goals"],
        observed_scales=observed_scales,
        prediction_scales=prediction_scales,
    )
    # (P, M=1, G, G) -> (P, G, G)
    return MethodPredictions(grids=np.asarray(grids[:, 0]), logp=np.asarray(logp[:, 0]))


def predict_sqmc_from_history(dataset, params, cfg, result, scales):
    """One-step-ahead SQMC grids from an already-run evaluation filter history.

    ``result`` is the dict returned by ``run_filter_sqmc`` (keys
    ``particles_x``, ``log_weights``) and ``scales`` is the ``(T, 1)`` per-match
    scale array used for that filter. This adapter slices the prediction rows
    and scales, invokes the extracted forecast helper, and builds
    ``MethodPredictions``. It shares the single evaluation filter with the
    ranking summaries, so predictions and rankings describe the same particles.
    """
    rng = pred_index_range(dataset)
    prediction = _slice_sqmc(dataset.sqmc, rng.start, dataset.sqmc.timestamp.shape[0])
    prediction_scales = scales[rng.start:]
    grids, logp, _ = predict_from_sqmc_history(
        result, prediction, rng.start, params["model"],
        cfg["max_goals"], prediction_scales,
    )
    return MethodPredictions(grids=np.asarray(grids[:, 0]), logp=np.asarray(logp[:, 0]))


def predict_ekf(dataset, params, cfg):
    """One-step-ahead EKF score grids via OU propagation + GH quadrature.

    ``run_filter`` stores ``history[i]`` = the marginal *before* assimilating
    match ``i``, so forecasting from ``history[i]`` never sees match ``i``'s
    result or any later one.
    """
    num_teams = len(dataset.teams)
    match_scale = cfg.get("match_scale", 1.0)
    history = ekf.run_filter(dataset.inputs, params, num_teams, match_scale=match_scale)

    rng = pred_index_range(dataset)
    grids = ekf.sequential_predict(
        dataset.inputs,
        history,
        params,
        rng.start,
        max_goals=cfg["max_goals"],
        degree=cfg.get("gauss_hermite_degree", 32),
        match_scale=match_scale,
    )
    grids = np.asarray(grids)
    # GH integrates the unnormalized likelihood mass; normalize to a distribution.
    grids = grids / grids.sum(axis=(-2, -1), keepdims=True)
    return MethodPredictions(grids=grids, logp=_grid_logp(grids, dataset, cfg))


def _grid_logp(grids, dataset, cfg):
    """Log-probability of each actual score under the score grid."""
    rng = pred_index_range(dataset)
    home_sc = np.asarray(dataset.inputs.score[rng, 0]).astype(int)
    away_sc = np.asarray(dataset.inputs.score[rng, 1]).astype(int)
    max_h = cfg["max_goals"]
    known = (home_sc >= 0) & (away_sc >= 0) & (home_sc <= max_h) & (away_sc <= max_h)
    logp = np.zeros(grids.shape[0])
    if known.any():
        idx = np.flatnonzero(known)
        logp[known] = np.log(grids[idx, home_sc[idx], away_sc[idx]] + 1e-12)
    return logp

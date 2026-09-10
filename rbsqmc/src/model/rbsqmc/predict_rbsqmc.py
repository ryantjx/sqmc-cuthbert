"""Sequential one-step-ahead prediction with the RB-SQMC filter."""

from functools import partial

import jax
import jax.numpy as jnp

from rbsqmc.src.data.data import concat_football_results, unpack_football_results
from rbsqmc.src.model.rbsqmc.model_rbsqmc import run_filter_sqmc
from rbsqmc.src.model.rbsmc.predict import predict_match_score
from rbsqmc.src.utils.type import EMParams, FootballResults


def run_sequential_predict_rbsqmc(
    key: jax.Array,
    observed_inputs: FootballResults,
    prediction_inputs: FootballResults,
    params: EMParams,
    n_particles: int,
    max_goals: int,
    observed_scales: jax.Array | None = None,
    prediction_scales: jax.Array | None = None,
):
    """Unpack to match-level steps, then run compiled SQMC prediction.

    ``observed_scales`` / ``prediction_scales`` are optional ``(T, 1)``
    per-match observation scales aligned with the (packed) inputs. When
    omitted they default to ones, preserving the behaviour of existing callers.
    """
    observed = unpack_football_results(observed_inputs)
    prediction = unpack_football_results(prediction_inputs)
    observed_scales = _unpack_scales(observed_inputs, observed, observed_scales)
    prediction_scales = _unpack_scales(prediction_inputs, prediction, prediction_scales)
    return _run_sequential_predict_rbsqmc_jitted(
        key=key,
        observed_inputs=observed,
        prediction_inputs=prediction,
        params=params,
        n_particles=n_particles,
        max_goals=max_goals,
        observed_scales=observed_scales,
        prediction_scales=prediction_scales,
    )


def _unpack_scales(packed, unpacked, scales):
    """Align packed ``(T, M)`` scales with the unpacked one-match-per-row order.

    ``unpack_football_results`` selects valid ``(t, m)`` entries via
    ``np.nonzero(match_mask)``. Scales must be unpacked with the same valid
    row/column indices so scale/match correspondence survives padding and
    multiple matches per date. When ``scales`` is None, return ones.

    Supported shapes are ``(T,)``, ``(T, 1)``, and ``(T, M)`` (matching the
    packed match mask). Unsupported shapes raise rather than relying on JAX
    index clipping. Valid entries must be finite and positive; masked padding
    may remain zero.
    """
    if scales is None:
        return jnp.ones((unpacked.timestamp.shape[0], 1), dtype=float)
    scales = jnp.asarray(scales)
    T, M = packed.match_mask.shape
    if scales.ndim == 1:
        if scales.shape[0] != T:
            raise ValueError(f"1-D scales must have length T={T}, got {scales.shape[0]}")
        scales = scales[:, None]
    elif scales.ndim == 2:
        if scales.shape[0] != T:
            raise ValueError(f"2-D scales must have T={T} rows, got {scales.shape[0]}")
        if scales.shape[1] not in (1, M):
            raise ValueError(f"2-D scales must have 1 or M={M} columns, got {scales.shape[1]}")
    else:
        raise ValueError(f"Unsupported scales shape {scales.shape}; expected (T,), (T, 1), or (T, M)")
    # Broadcast a one-column scale to (T, M) after both accepted input-shape
    # branches, so (T,) and (T, 1) are handled identically and never rely on
    # JAX index clipping for multi-match rows.
    if scales.shape[1] == 1 and M != 1:
        scales = jnp.broadcast_to(scales, (T, M))
    valid_t, valid_m = jnp.nonzero(jnp.asarray(packed.match_mask))
    selected = scales[valid_t, valid_m]
    if not jnp.all(jnp.isfinite(selected)) or not jnp.all(selected > 0):
        raise ValueError("Scales on valid matches must be finite and positive")
    return selected[:, None]


@partial(jax.jit, static_argnames=("n_particles", "max_goals"))
def _run_sequential_predict_rbsqmc_jitted(
    key: jax.Array,
    observed_inputs: FootballResults,
    prediction_inputs: FootballResults,
    params: EMParams,
    n_particles: int,
    max_goals: int,
    observed_scales: jax.Array,
    prediction_scales: jax.Array,
):
    """Run SQMC and return one-step-ahead score forecasts.

    The full observed-plus-prediction sequence is filtered once. For prediction
    timestep ``t``, the forecast uses the particles at history index ``t+1``:
    their positions have been propagated for the current match, while their
    score-dependent weights are replaced by uniform pre-likelihood weights.
    Prediction inputs should contain one match per leading timestep so
    same-date matches are forecast and updated sequentially.
    """
    full_inputs = concat_football_results(observed_inputs, prediction_inputs)
    full_scales = jnp.concatenate([observed_scales, prediction_scales], axis=0)
    result, _ = run_filter_sqmc(
        key=key,
        model_inputs=full_inputs,
        params=params,
        n_particles=n_particles,
        max_goals=max_goals,
        match_scales=full_scales,
    )
    return predict_from_sqmc_history(
        result, prediction_inputs, observed_inputs.timestamp.shape[0],
        params, max_goals, prediction_scales,
    )


def predict_from_sqmc_history(
    result,
    prediction_inputs: FootballResults,
    prediction_start: int,
    params: EMParams,
    max_goals: int,
    prediction_scales: jax.Array,
):
    """One-step-ahead score forecasts from an already-run filter history.

    This helper must *not* call ``run_filter_sqmc``; it consumes a filter
    ``result`` dict (keys ``particles_x``, ``log_weights``) and produces the
    same three-array contract as the sequential driver. It requires one valid
    match per row in ``prediction_inputs`` (a history that stores only an
    end-of-day state cannot supply pre-likelihood particles for every match
    within that day).

    State/weight contract (bootstrap filter: positions are generated before the
    likelihood is applied, and the current score changes only the weights):

    | Quantity | Coordinates | Weights |
    | --- | --- | --- |
    | Previous filtered state, before propagation to match ``t`` | ``particles_x[t]`` | ``log_weights[t]`` |
    | Forecast at match ``t``, before its score is assimilated | ``particles_x[t + 1]`` | Uniform |
    | Posterior after match ``t`` | ``particles_x[t + 1]`` | ``log_weights[t + 1]`` |

    For prediction row ``p``, ``t = prediction_start + p`` and the forecast uses
    ``particles_x[t + 1]`` with uniform predictive weights and the current
    fixture's scale ``prediction_scales[p, 0]``.
    """
    particles_x = result["particles_x"]
    log_weights = result["log_weights"]
    grid_size = max_goals + 1

    # Enforce the one-match-per-row contract and history bounds before the
    # compiled scan. A history that stores only an end-of-day state cannot
    # supply pre-likelihood particles for every match within that day. Only
    # static (shape/index) checks are performed here so the helper remains
    # usable inside a jitted caller without coercing tracers to NumPy.
    n_prediction_steps = prediction_inputs.timestamp.shape[0]
    if prediction_inputs.match_mask.shape[1] != 1:
        raise ValueError("predict_from_sqmc_history requires one match per row")
    if prediction_start < 0 or prediction_start + n_prediction_steps + 1 > particles_x.shape[0]:
        raise ValueError("prediction_start out of range for the supplied filter history")
    # The scale array must be exactly (P, 1), one scale per prediction step.
    if prediction_scales.ndim != 2 or prediction_scales.shape != (n_prediction_steps, 1):
        raise ValueError(
            f"prediction_scales must have shape ({n_prediction_steps}, 1), got {prediction_scales.shape}"
        )
    # Particle and weight history dimensions must be compatible.
    if particles_x.ndim != 4 or log_weights.ndim != 2:
        raise ValueError("filter history particles_x must be (T+1, N, teams, 2) and log_weights (T+1, N)")
    if particles_x.shape[0] != log_weights.shape[0]:
        raise ValueError("filter history particles_x and log_weights must have the same time length")
    if particles_x.shape[1] != log_weights.shape[1]:
        raise ValueError("filter history particles_x and log_weights must have the same particle count")

    def scan_body(_, prediction_step):
        history_index = prediction_start + prediction_step + 1
        particles_predictive = particles_x[history_index]
        weights_predictive = jnp.zeros_like(log_weights[history_index])
        scale = prediction_scales[prediction_step, 0]

        home_ids = prediction_inputs.matches.home_id[prediction_step]
        away_ids = prediction_inputs.matches.away_id[prediction_step]
        valid = prediction_inputs.match_mask[prediction_step]
        home_scores = prediction_inputs.matches.home_score[prediction_step]
        away_scores = prediction_inputs.matches.away_score[prediction_step]

        def predict_one(home_id, away_id, is_valid, home_score, away_score):
            grid = predict_match_score(
                particles_predictive,
                weights_predictive,
                home_id,
                away_id,
                params.alpha,
                params.beta,
                max_goals,
                scale,
            )
            grid = jnp.where(
                is_valid,
                grid,
                jnp.zeros((grid_size, grid_size)),
            )
            score_is_known = (
                is_valid
                & (home_score >= 0)
                & (away_score >= 0)
                & (home_score <= max_goals)
                & (away_score <= max_goals)
            )
            safe_home_score = jnp.clip(home_score, 0, max_goals)
            safe_away_score = jnp.clip(away_score, 0, max_goals)
            log_probability = jnp.where(
                score_is_known,
                jnp.log(grid[safe_home_score, safe_away_score] + 1e-12),
                0.0,
            )
            return grid, log_probability

        grids, log_probabilities = jax.vmap(predict_one)(
            home_ids,
            away_ids,
            valid,
            home_scores,
            away_scores,
        )
        return None, (
            grids,
            log_probabilities,
            jnp.sum(log_probabilities),
        )

    _, outputs = jax.lax.scan(
        scan_body,
        None,
        jnp.arange(n_prediction_steps),
    )
    return outputs


__all__ = ["run_sequential_predict_rbsqmc", "predict_from_sqmc_history"]

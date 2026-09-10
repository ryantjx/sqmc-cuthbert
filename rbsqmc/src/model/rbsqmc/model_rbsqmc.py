"""
Rao-Blackwellized SQMC (RB-SQMC) model for football match modelling.

This module implements the RB-SQMC algorithm described in Chapter 2 of the
dissertation.  The key difference from the RB-SMC filter in ``model.py`` is
that the propagation step is deterministic (inverse-CDF transform driven by
QMC points) and the resampling step uses Hilbert-sorted inverse-CDF selection
instead of stochastic systematic resampling.

The algorithm at each time step t:

1. Generate N RQMC points (u_t, v_t) of dimension 1 + d = 5
   - u_t: first coordinate, used for ancestor selection
   - v_t: remaining d=4 coordinates, used for deterministic propagation

2. Hilbert sort the previous N particles on the observed coordinates

3. Resampling by inverse transform: sort QMC by first coordinate, select
   ancestors via inverse CDF using Hilbert-ordered weights

4. Propagate: deterministic transform x_t^O = mu_pred + L * Phi^{-1}(v_t)
   Recover complementary coordinates via Kalman conditioning

5. Compute log-weights from the bivariate Poisson likelihood
"""

import os
import jax
import jax.numpy as jnp
from functools import partial
from typing import cast

from rbsqmc.src.data.bivariate_poisson import loglik
from rbsqmc.src.data.data import get_results, WORLDCUP_2026_TEAMS
from rbsqmc.src.utils.helpers import (
    default_init_params,
    generate_rbpf_trajectory,
    _ensure_symmetric,
    _scale_aware_jitter,
)
from rbsqmc.src.utils.type import (
    RBPFFootballResults,
    EMParams,
    FootballResults,
)
from rbsqmc.src.model.rbsmc.model import compute_gamma_trajectory

# SQMC building blocks
from sqmc.hilbert_sort.hilbert_sort import hilbert_sort
from sqmc.qmc.qmc import Sobol, normal_coordinates

# Default to CPU locally, but allow the GPU pipeline to force a device via
# the RBSQMC_PLATFORM env var (e.g. RBSQMC_PLATFORM=cuda on a Colab T4).
jax.config.update(
    "jax_platforms", os.environ.get("RBSQMC_PLATFORM", "cpu")
)

# ---------------------------------------------------------------------------
# QMC point generation
# ---------------------------------------------------------------------------

def generate_rqmc_points(
    key: jax.Array,
    n: int,
    d: int,
) -> jnp.ndarray:
    """Generate N randomized QMC (RQMC) points of dimension d.

    Uses a scrambled Sobol sequence so that each call produces an independent
    randomised low-discrepancy point set.  The scrambling key changes every
    call, giving independent RQMC replicates across filter runs.

    Args:
        key: JAX PRNG key for scrambling.
        n: Number of points.
        d: Dimension of each point.

    Returns:
        Array of shape (n, d) in [0, 1]^d.
    """
    # A fresh scrambled engine is constructed from ``key`` so each call
    # produces an independent randomization starting at index zero, matching
    # the generic SQMC filter in ``sqmc/sqmc/sqmc.py``. ``start_index=0``
    # keeps the point set a balanced net (Owen, 2020); the open-interval
    # policy for the Gaussian quantile is handled by ``normal_coordinates``.
    sobol = Sobol(
        d=d,
        scramble=True,
        key=key,
        dtype=jnp.float64,
        start_index=0,
    )
    # ``sample`` is typed as returning ``(points, next_state)`` when an
    # explicit ``state`` is passed; without one it returns only ``points``.
    return cast(jnp.ndarray, sobol.sample(n))


# ---------------------------------------------------------------------------
# RB-SQMC filter step
# ---------------------------------------------------------------------------


def _log_match_potential_batched(
    particles_x: jnp.ndarray,
    home_id: jax.Array,
    away_id: jax.Array,
    home_score: jax.Array,
    away_score: jax.Array,
    alpha: float,
    beta: float,
    max_goals: int,
    scale: jax.Array = 1.0,
) -> jnp.ndarray:
    """Return one match's log-likelihood for every particle."""
    known = (home_score >= 0) & (away_score >= 0)
    # Keep both branches on the likelihood's support so missing observations
    # cannot create undefined gamma-function derivatives.
    y = jnp.where(known, jnp.array([home_score, away_score]), jnp.zeros(2, dtype=int))
    x_i = particles_x[:, home_id, :]
    x_j = particles_x[:, away_id, :]
    potential = jax.vmap(
        lambda xi, xj: loglik(
            y,
            xi,
            xj,
            alpha=alpha,
            beta=beta,
            max_goals=max_goals,
            scale=scale,
        )
    )(x_i, x_j)
    return jnp.where(known, potential, 0.0)


def propagate_match_transform(
    particles_x: jnp.ndarray,
    B: jnp.ndarray,
    kalman_gain: jnp.ndarray,
    home_id: jax.Array,
    away_id: jax.Array,
    gamma_observed: jnp.ndarray,
    v_t: jnp.ndarray,
    n_particles: int,
) -> jnp.ndarray:
    """Propagate one match with its own four-dimensional QMC coordinates.

    ``particles_x`` is already the OU-predicted and resampled particle array.
    The sampled match coordinates are then used to condition every team's
    Rao--Blackwellized mean through the precomputed Kalman gain.
    """
    obs_indices = jnp.array([home_id, away_id])
    mu_O = particles_x[:, obs_indices, :]

    Sigma_OO = jnp.kron(gamma_observed, B)
    Sigma_OO = _ensure_symmetric(
        Sigma_OO + _scale_aware_jitter(Sigma_OO) * jnp.eye(4)
    )
    L_t = jnp.linalg.cholesky(Sigma_OO)

    # ``normal_coordinates`` applies the 30-bit Sobol' open-interval policy
    # (clip to [2^-31, 1 - 2^-31] and nextafter(1, 0)) before the Gaussian
    # quantile, so exact-zero and rounded-one RQMC coordinates stay finite.
    z = normal_coordinates(v_t)
    x_O_flat = mu_O.reshape(n_particles, 4) + z @ L_t.T
    x_O = x_O_flat.reshape(n_particles, 2, 2)

    diff = x_O - mu_O
    x_updated = particles_x + jnp.einsum(
        "tk,nkd->ntd", kalman_gain, diff
    )
    return x_updated.at[:, obs_indices, :].set(x_O)


# ---------------------------------------------------------------------------
# Hilbert-sorted inverse-CDF resampling (Steps 2-3 of RB-SQMC)
# ---------------------------------------------------------------------------

def hilbert_resample(
    particles_x: jnp.ndarray,  # (N, num_teams, 2)
    log_weights: jnp.ndarray,  # (N,)
    rqmc_points: jnp.ndarray,  # (N, 5) — complete (u, v) points
    obs_indices: jnp.ndarray,  # (2,) — team indices involved in the current match
    n_particles: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Hilbert-sorted inverse-CDF ancestor selection.

    1. Extract the observed coordinates (2 teams × 2 dims = 4) from each
       particle and flatten to (N, 4).
    2. Hilbert-sort the particles on these observed coordinates.
    3. Build the cumulative distribution from the Hilbert-ordered weights.
    4. Sort the QMC first coordinates and select ancestors via inverse CDF.

    Returns:
        ``(indices, v_sorted)`` where ``indices`` selects ancestors from the
        original particle array and ``v_sorted`` retains the four propagation
        coordinates paired with each sorted first coordinate.
    """
    # Extract observed coordinates: (N, 2, 2) → (N, 4)
    x_obs = particles_x[:, obs_indices, :]  # (N, 2, 2)
    x_flat = x_obs.reshape(n_particles, -1)  # (N, 4)

    # Hilbert sort on the 4-dimensional observed coordinates
    sort_idx = hilbert_sort(x_flat)  # (N,) permutation

    # Reorder weights
    w = jnp.exp(log_weights - jax.scipy.special.logsumexp(log_weights))
    w_sorted = w[sort_idx]  # (N,)

    # Cumulative distribution
    W = jnp.cumsum(w_sorted)  # (N,)

    # Sort complete QMC points, preserving each u/v pairing.
    qmc_order = jnp.argsort(rqmc_points[:, 0], stable=True)
    rqmc_sorted = rqmc_points[qmc_order]
    u_sorted = rqmc_sorted[:, 0]

    # Inverse CDF: for each u, find the first j where W[j] >= u
    ancestors_sorted = jnp.searchsorted(W, u_sorted, side="left")
    ancestors_sorted = jnp.minimum(ancestors_sorted, n_particles - 1)

    # Map back to original indices
    ancestors = sort_idx[ancestors_sorted]  # (N,)

    return ancestors, rqmc_sorted[:, 1:]


def _differentiable_resampled_log_weights(
    log_weights: jnp.ndarray,
    ancestors: jnp.ndarray,
    n_particles: int,
) -> jnp.ndarray:
    """Uniform forward weights with the resampling score in the gradient.

    This is the log-space form of the differentiable particle-filter ratio
    ``w[a] / stop_gradient(w[a])``. Its numerical value is one, while its
    derivative preserves the parameter dependence of the resampling weights.
    """
    selected_log_w = log_weights[ancestors]
    return (
        -jnp.log(n_particles)
        + selected_log_w
        - jax.lax.stop_gradient(selected_log_w)
    )


# ---------------------------------------------------------------------------
# RB-SQMC filter (full forward pass)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("n_particles", "max_goals"))
def run_filter_sqmc(
    key: jax.Array,
    model_inputs: FootballResults,
    params: EMParams,
    n_particles: int,
    max_goals: int,
    *,
    match_scales: jax.Array | None = None,
) -> tuple[dict, RBPFFootballResults]:
    """Run the RB-SQMC forward filter.

    Implements the 5-step RB-SQMC algorithm from the dissertation:

    1. Generate N RQMC points of dimension 1 + d = 5
    2. Hilbert sort previous N particles
    3. Resampling by inverse transform (ancestor selection)
    4. Propagate via deterministic transform + Kalman conditioning
    5. Compute log-weights

    Returns:
        A dict with keys:
            'particles_x': (T+1, N, num_teams, 2) — particle states
            'log_weights': (T+1, N) — log particle weights
            'log_normalizing_constant': (T+1,) — accumulated log Z
        and the augmented model_inputs_rbpf.
    """
    _, filter_key = jax.random.split(key)
    if match_scales is None:
        match_scales = jnp.ones_like(model_inputs.match_mask, dtype=float)

    # Precompute the deterministic Gamma trajectory (shared across particles)
    gamma, gamma_pred, gamma_observed, kalman_gain = compute_gamma_trajectory(
        model_inputs=model_inputs,
        gamma_0=params.gamma_0,
        kappa=params.kappa,
        num_teams=params.mean_0.shape[0],
    )
    model_inputs_rbpf = generate_rbpf_trajectory(
        model_inputs=model_inputs,
        gamma=gamma,
        gamma_pred=gamma_pred,
        gamma_observed=gamma_observed,
        kalman_gain=kalman_gain,
    )

    num_teams = params.mean_0.shape[0]
    d = 4  # effective dimension (2 teams × 2 coordinates)
    qmc_dim = 1 + d  # 5

    T = model_inputs_rbpf.timestamp.shape[0]

    # Initialise particles at the prior mean
    x_init = jnp.broadcast_to(
        params.mean_0, (n_particles, num_teams, 2)
    ).astype(jnp.float64)
    log_w_init = jnp.full((n_particles,), -jnp.log(n_particles))

    # Log-normalizing constant
    log_z = jnp.array(0.0)

    def filter_step(carry, t):
        x_prev, log_w_prev, log_z_prev, key = carry

        model_input_t = jax.tree.map(lambda x: x[t], model_inputs_rbpf)

        # Apply the OU prediction once per date. Matches on that date are then
        # assimilated sequentially with dt=0, so every update remains a
        # five-dimensional SQMC step even when a date contains many matches.
        dt = model_input_t.timestamp - model_input_t.timestamp_prev
        phi_t = jnp.exp(-params.kappa * dt)
        x_pred = params.mean_0 + phi_t * (x_prev - params.mean_0)

        def match_step(match_carry, match):
            x, log_w, log_z, key = match_carry
            K, home_id, away_id, gamma_OO, home_score, away_score, valid, scale = match

            def update_valid(valid_carry):
                x, log_w, log_z, key = valid_carry

                # --- Steps 1--3: paired RQMC generation and resampling ---
                key, sobol_key = jax.random.split(key)
                rqmc_points = generate_rqmc_points(
                    key=sobol_key,
                    n=n_particles,
                    d=qmc_dim,
                )
                obs_indices = jnp.array([home_id, away_id])
                ancestors, v_t = hilbert_resample(
                    particles_x=x,
                    # The categorical/Hilbert resampling decision is discrete.
                    log_weights=jax.lax.stop_gradient(log_w),
                    rqmc_points=rqmc_points,
                    obs_indices=obs_indices,
                    n_particles=n_particles,
                )

                x_resampled = x[ancestors]

                # Numerically these are uniform weights. The ratio
                # w / stop_gradient(w) implements the differentiable particle
                # filter correction while leaving the forward pass unchanged.
                log_w_resampled = _differentiable_resampled_log_weights(
                    log_weights=log_w,
                    ancestors=ancestors,
                    n_particles=n_particles,
                )

                # --- Step 4: propagate this match with its paired v point ---
                x_propagated = propagate_match_transform(
                    particles_x=x_resampled,
                    B=params.B,
                    kalman_gain=K,
                    home_id=home_id,
                    away_id=away_id,
                    gamma_observed=gamma_OO,
                    v_t=v_t,
                    n_particles=n_particles,
                )

                # --- Step 5: weight this observation ---
                log_potential = _log_match_potential_batched(
                    particles_x=x_propagated,
                    home_id=home_id,
                    away_id=away_id,
                    home_score=home_score,
                    away_score=away_score,
                    alpha=params.alpha,
                    beta=params.beta,
                    max_goals=max_goals,
                    scale=scale,
                )
                log_w_unnorm = log_w_resampled + log_potential
                log_z_new = log_z + jax.scipy.special.logsumexp(log_w_unnorm)
                log_w_norm = (
                    log_w_unnorm
                    - jax.scipy.special.logsumexp(log_w_unnorm)
                )
                return x_propagated, log_w_norm, log_z_new, key

            return jax.lax.cond(valid, update_valid, lambda c: c, match_carry), None

        match_inputs = (
            model_input_t.kalman_gain,
            model_input_t.matches.home_id,
            model_input_t.matches.away_id,
            model_input_t.gamma_observed,
            model_input_t.matches.home_score,
            model_input_t.matches.away_score,
            model_input_t.match_mask,
            match_scales[t],
        )
        (x_new, log_w_new, log_z_new, key), _ = jax.lax.scan(
            match_step,
            (x_pred, log_w_prev, log_z_prev, key),
            match_inputs,
        )

        return (x_new, log_w_new, log_z_new, key), (
            x_new, log_w_new, log_z_new,
        )

    # Scan over time steps
    carry = (x_init, log_w_init, log_z, filter_key)
    _, (particles_x, log_weights, log_z_history) = jax.lax.scan(
        f=filter_step,
        init=carry,
        xs=jnp.arange(T),
    )

    # Prepend initial state
    particles_x_full = jnp.concatenate(
        [x_init[None], particles_x], axis=0
    )  # (T+1, N, num_teams, 2)
    log_weights_full = jnp.concatenate(
        [log_w_init[None], log_weights], axis=0
    )  # (T+1, N)
    log_z_full = jnp.concatenate(
        [jnp.array(0.0)[None], log_z_history], axis=0
    )  # (T+1,)

    result = {
        'particles_x': particles_x_full,
        'log_weights': log_weights_full,
        'log_normalizing_constant': log_z_full,
    }
    return result, model_inputs_rbpf


# ---------------------------------------------------------------------------
# Convenience: run filter and return log Z (for optimisation)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("n_particles", "max_goals"))
def run_filter_sqmc_logz(
    key: jax.Array,
    model_inputs: FootballResults,
    params: EMParams,
    n_particles: int,
    max_goals: int,
    *,
    match_scales: jax.Array | None = None,
) -> jnp.ndarray:
    """Run the RB-SQMC filter and return the final log-normalizing constant."""
    result, _ = run_filter_sqmc(
        key=key,
        model_inputs=model_inputs,
        params=params,
        n_particles=n_particles,
        max_goals=max_goals,
        match_scales=match_scales,
    )
    return result['log_normalizing_constant'][-1]


def main(
    start_date: str = "1980-01-01",
    end_date: str = "2026-01-01",
    n_particles: int = 10,
    max_goals: int = 8,
    seed: int = 42,
) -> tuple[dict, RBPFFootballResults]:
    """Run one RB-SQMC pass using the standard model configuration."""
    data, model_inputs, team_id_to_name = get_results(
        start_date=start_date,
        end_date=end_date,
        max_goals=max_goals,
        include_friendly=False,
        teams_only=WORLDCUP_2026_TEAMS,
        download=False,
    )
    num_teams = len(team_id_to_name)
    params = default_init_params(
        num_teams=num_teams,
        team_id_to_name=team_id_to_name,
    )

    print(
        f"Running RB-SQMC from {start_date} to {end_date}: "
        f"{num_teams} teams, {len(data)} dates, "
        f"{int(model_inputs.match_mask.sum())} matches, "
        f"{n_particles} particles."
    )
    result, augmented_inputs = run_filter_sqmc(
        key=jax.random.PRNGKey(seed),
        model_inputs=model_inputs,
        params=params,
        n_particles=n_particles,
        max_goals=max_goals,
    )

    # The filter is JIT-compiled and asynchronous; synchronizing here ensures
    # the command reports success only after the complete pass has finished.
    result["log_normalizing_constant"].block_until_ready()
    print(
        "RB-SQMC completed: "
        f"particles={result['particles_x'].shape}, "
        f"final log Z={float(result['log_normalizing_constant'][-1]):.6f}."
    )
    return result, augmented_inputs


if __name__ == "__main__":
    main()

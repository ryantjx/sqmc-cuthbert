import os
import jax
import jax.numpy as jnp
import cuthbert
import cuthbertlib
from functools import partial

from rbsqmc.src.data.bivariate_poisson import loglik
from rbsqmc.src.data.data import get_results, WORLDCUP_2026_TEAMS
from rbsqmc.src.utils.helpers import (
    default_init_params,
    generate_rbpf_trajectory,
    _ensure_symmetric,
    _scale_aware_jitter,
)
from rbsqmc.src.utils.type import RBPFState, RBPFFootballResults, EMParams, FootballResults

# Default to CPU locally, but allow the GPU pipeline to force a device via
# the RBSQMC_PLATFORM env var (e.g. RBSQMC_PLATFORM=cuda on a Colab T4).
jax.config.update(
    "jax_platforms", os.environ.get("RBSQMC_PLATFORM", "cpu")
)

MAX_GOALS = 8

@partial(jax.jit, static_argnames=("num_teams",))
def compute_gamma_trajectory(
    model_inputs: FootballResults, 
    gamma_0: jnp.ndarray, 
    kappa: float,
    num_teams: int
):
    """
    Compute deterministic transition of covariance matrix Gamma over time using the OU process.
    
    works for multiple matches per timestamp, but does not work if a team plays two matches on a single day
    """
    # carry, xs
    def kalman_update(gamma_prev, model_input):
        dt = model_input.timestamp - model_input.timestamp_prev
        phi_t = jnp.exp(-kappa * dt)
        Q = (1 - phi_t**2) * gamma_0
        gamma_pred = _ensure_symmetric(phi_t**2 * gamma_prev + Q)
        # gamma_pred = phi_t**2 * gamma_prev + Q
        
        home_id = model_input.matches.home_id # (M, ) matches per timestamp
        away_id = model_input.matches.away_id # (M, )
        valid = model_input.match_mask # (M, )

        def kalman_update_single(gamma_current, match):
            # at each timestamp, I will have m matches (m < M = max number of matches per day)
            home_id, away_id, valid = match
            def update_valid(gamma_current):
                obs_indices = jnp.array([home_id, away_id])

                gamma_OO = gamma_current[jnp.ix_(obs_indices, obs_indices)]
                gamma_RO = gamma_current[:, obs_indices]
                # Scale-aware jitter so the 2x2 pinv is strictly positive-
                # definite. Without it, GPU float32 rounding can leave gamma_OO
                # with a tiny negative eigenvalue, making pinv / the Cholesky in
                # the downstream sampler return NaN. The jitter is small relative
                # to the matrix scale, so it does not perturb the Kalman gain.
                gamma_OO = _ensure_symmetric(
                    gamma_OO + _scale_aware_jitter(gamma_OO) * jnp.eye(2)
                )
                K = gamma_RO @ jnp.linalg.pinv(gamma_OO)

                # Kalman update: sequentially update gamma_current for each match
                gamma_updated = _ensure_symmetric(gamma_current - K @ gamma_RO.T)
                # gamma_updated = gamma_current - K @ gamma_RO.T
                # Zero observed teams' rows/cols (Schur marginalization).
                obs_mask = jnp.zeros(num_teams, dtype=bool).at[obs_indices].set(True)
                keep_mask = jnp.outer(~obs_mask, ~obs_mask)
                gamma_updated = gamma_updated * keep_mask
                return (gamma_updated, (gamma_OO, K))

            # Padded (invalid) matches pass through unchanged.
            def skip_padding(gamma_current):
                return (gamma_current, (jnp.zeros((2, 2), dtype=gamma_current.dtype), jnp.zeros((num_teams, 2), dtype=gamma_current.dtype)))
            # gamma_new = jnp.where(valid, gamma_updated, gamma_current)
            # Zero out the Kalman gain for invalid matches to avoid NaNs in the log-likelihood.
            # K = jnp.where(valid, K, jnp.zeros_like(K))
            # return 
            # if valid, we reun the kalman_update_single, otherwise we skip and return the current gamma unchanged, with zeroed out kalman gain and observed gamma
            return jax.lax.cond(
                valid,
                update_valid,
                skip_padding,
                gamma_current,
            )
        
        gamma_updated, (gamma_observed, kalman_gain) = jax.lax.scan(
            f=kalman_update_single,
            init=gamma_pred,
            xs=(home_id, away_id, valid)
        )
        # gamma_updated: (M, M) final carry (day posterior),
        # kalman_gain: (M_max, M, 2) per-match gains
        return gamma_updated, (gamma_updated, gamma_pred, gamma_observed, kalman_gain)
    
    _, (gamma_updated, gamma_pred, gamma_observed, kalman_gain) = jax.lax.scan(
        f=kalman_update, 
        init=gamma_0, 
        xs=model_inputs,
    )
    return gamma_updated, gamma_pred, gamma_observed, kalman_gain

def init_sample(
    key: jax.Array,
    model_inputs: FootballResults,
    mean_0: jnp.ndarray,
    # gamma_0: jnp.ndarray,
    # B: jnp.ndarray
) -> RBPFState:
    """
    draw from initial parameters
    """
    # num_teams = mean_0.shape[0]
    # K = mean_0.shape[1]
    # x = jax.random.multivariate_normal(
    #     key, mean_0.flatten(), jnp.kron(gamma_0, B)
    # ).reshape(num_teams, K)
    # return RBPFState(x=x)
    return RBPFState(x=mean_0)  # deterministic initialization for testing

def propagate_sample(
    key: jax.Array,
    state: RBPFState,
    model_inputs: RBPFFootballResults,
    mean: jnp.ndarray,
    B: jnp.ndarray,
    kappa: float,
):
    """Propagate the state forward in time via the OU process."""
    # 1. OU prediction: one latent step per day
    dt = model_inputs.timestamp - model_inputs.timestamp_prev
    phi_t = jnp.exp(-kappa * dt)
    pred_mean = mean + phi_t * (state.x - mean)

    kalman_gain = model_inputs.kalman_gain # # (M, N, 2) single day kalman gain for each match, where M = max matches per day, N = number of teams

    home_id = model_inputs.matches.home_id # (M, ) matches per timestamp
    away_id = model_inputs.matches.away_id # (M, ) matches per timestamp
    gamma_observed = model_inputs.gamma_observed # (T, M x M)
    valid = model_inputs.match_mask # (T, M) boolean mask indicating valid matches

    def match_step(carry, match):
        x, key = carry
        K, home_id, away_id, gamma_OO, valid = match

        # skip padded matches (invalid) and return the current state unchanged
        def update_valid(carry):
            x, key = carry
            obs_indices = jnp.array([home_id, away_id])
            mu_O = x[obs_indices]  # (2, 2)
            Sigma_OO = jnp.kron(gamma_OO, B)  # (4, 4)

            key, subkey = jax.random.split(key)
            # Scale-aware jitter on the 4x4 sampling covariance so the Cholesky
            # inside `multivariate_normal` is strictly positive-definite. This
            # guards against the rare case where `gamma_OO` is singular (a
            # team's posterior variance was zeroed) or has tiny negative
            # eigenvalues from GPU float32 rounding, which would make the
            # multivariate-normal sampler return NaN. Zero-variance directions
            # stay essentially at the mean.
            Sigma_OO = _ensure_symmetric(
                Sigma_OO + _scale_aware_jitter(Sigma_OO) * jnp.eye(4)
            )
            x_O_flat = jax.random.multivariate_normal(subkey, mu_O.reshape(-1), Sigma_OO)  # (4,)
            x_O = x_O_flat.reshape(2, 2)  # (2, 2)

            # kalman update for the new state x
            x_updated = x + K @ (x_O - mu_O)  # condition all teams
            x_updated = x_updated.at[obs_indices].set(x_O)  # observed = sampled
            return (x_updated, key), x_updated
        def skip_padding(carry):
            x, key = carry
            return (x, key), x
        return jax.lax.cond(
            valid,
            update_valid,
            skip_padding,
            carry
        )
    (x_new, _), _ = jax.lax.scan(
        f=match_step,
        init=(pred_mean, key),
        xs=(kalman_gain, home_id, away_id, gamma_observed, valid)
    )
    return RBPFState(x=x_new)


def _log_potential(
    state_prev: RBPFState,
    state: RBPFState,
    model_inputs: RBPFFootballResults,
    alpha: float,
    beta: float,
    max_goals: int,
) -> jax.Array:
    home_id = model_inputs.matches.home_id    # (M_max,)
    away_id = model_inputs.matches.away_id    # (M_max,)
    home_score = model_inputs.matches.home_score   # (M_max,)
    away_score = model_inputs.matches.away_score   # (M_max,)
    valid = model_inputs.match_mask           # (M_max,) bool

    def log_match(_, match):
        h, a, yh, ya, v = match
        y = jnp.array([yh, ya])                      # observed goals (2,)
        x_i = state.x[h]                             # (2,) home attack/defence
        x_j = state.x[a]                             # (2,) away attack/defence
        ll = loglik(y, x_i, x_j, alpha=alpha, beta=beta,
                    max_goals=max_goals, scale=1.0)
        return _, jnp.where(v, ll, 0.0)              # padded matches contribute 0

    _, logliks = jax.lax.scan(
        log_match, None, (home_id, away_id, home_score, away_score, valid)
    )
    return jnp.sum(logliks) 

def run_filter(
    key: jax.Array,
    model_inputs: FootballResults,
    params: EMParams,
    n_particles: int,
    max_goals: int,
) -> tuple[jax.Array, RBPFFootballResults]:
    """Unpack matches on the host, then run the compiled match-level filter.

    ``unpack_football_results`` uses NumPy to select the valid entries of the
    padded ``(date, match)`` arrays.  That dynamic shape operation cannot run
    while JAX is tracing, so it must remain outside the jitted implementation.
    """
    from rbsqmc.src.data.data import unpack_football_results

    model_inputs = unpack_football_results(model_inputs)
    return _run_filter_jitted(
        key=key,
        model_inputs=model_inputs,
        params=params,
        n_particles=n_particles,
        max_goals=max_goals,
    )


@partial(jax.jit, static_argnames=("n_particles", "max_goals"))
def _run_filter_jitted(
    key: jax.Array,
    model_inputs: FootballResults,
    params: EMParams,
    n_particles: int,
    max_goals: int,
) -> tuple[jax.Array, RBPFFootballResults]:
    key, init_key, filter_key = jax.random.split(key, 3)

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
        kalman_gain=kalman_gain
    )
    rbpf = cuthbert.smc.particle_filter.build_filter(
        init_sample=partial(
            init_sample,
            mean_0=params.mean_0,
            # gamma_0=params.gamma_0,
            # B=params.B
        ),
        propagate_sample=partial(
            propagate_sample,
            mean=params.mean_0,
            B=params.B,
            kappa=params.kappa,
        ),
        log_potential=partial(
            _log_potential,
            alpha=params.alpha,
            beta=params.beta,
            max_goals=max_goals,
        ),
        n_filter_particles=n_particles,
        resampling_fn=cuthbertlib.resampling.systematic.resampling,
    )

    # prepare initial state for the filter. use first value to prepare, but pass the full model_inputs_rbpf to the filter, which will handle the time steps and propagate the state accordingly.
    init_state = rbpf.init_prepare(model_inputs=jax.tree.map(lambda x: x[0], model_inputs_rbpf),
        key=init_key,
    )
    # since init_state is only used for preparing the filter, we pass the full model_inputs_rbpf to the filter, which will handle the time steps and propagate the state accordingly.
    filtered_states = cuthbert.filtering.filter(
        filter_obj=rbpf,
        model_inputs=model_inputs_rbpf,
        init_state=init_state,
        key=filter_key,
    )
    return filtered_states, model_inputs_rbpf


def main():
    ######################
    start_date = "2000-01-01"
    end_date = "2026-01-01"
    N = 10
    MAX_GOALS = 8

    #####################
    data, model_inputs, team_id_to_name = get_results(
        start_date=start_date,
        end_date=end_date,
        max_goals=MAX_GOALS,
        include_friendly=False,
        teams_only=WORLDCUP_2026_TEAMS,
    )
    num_teams = len(team_id_to_name)
    print(f"Extracted data from {start_date} to {end_date}, with {num_teams} teams and {len(data)} dates.")
    print("Number of teams:", num_teams)

    key = jax.random.PRNGKey(42)
    params = default_init_params(num_teams, team_id_to_name=team_id_to_name)

    # model_inputs = unpack_football_results(model_inputs)

    # gamma, gamma_pred, gamma_observed, kalman_gain = compute_gamma_trajectory(
    #     model_inputs=model_inputs,
    #     gamma_0=params.gamma_0,
    #     kappa=params.kappa,
    #     num_teams=params.mean_0.shape[0],
    # )
    # model_inputs_rbpf = generate_rbpf_trajectory(
    #     model_inputs=model_inputs,
    #     gamma=gamma,
    #     gamma_pred=gamma_pred,
    #     gamma_observed=gamma_observed,
    #     kalman_gain=kalman_gain
    # )
    # from rbsqmc.src.data.data import rbpf_inputs_to_dataframe
    # rbsqmc_df = rbpf_inputs_to_dataframe(model_inputs_rbpf, team_id_to_name=team_id_to_name)
    # rbsqmc_df.to_csv("./rbsqmc/rbpf_inputs.csv", index=False)
    key, filter_key = jax.random.split(key)

    print("Running filter (OU)...")
    try:
        filtered_states, model_inputs_rbpf = run_filter(
            key=filter_key,
            model_inputs=model_inputs,
            params=params,
            n_particles=N,
            max_goals=MAX_GOALS,
        )
    except Exception as e:
        print("Error during filtering:", e)
        return
    print("Filter completed successfully.")
    # print("Starting filtered states: ", filtered_states.particles.x[0])
    # print("Ending filtered states: ", filtered_states.particles.x[-1])
    # Print results of final filtered shape
    # Analyze the log-normalizing constant of the filter
    # 1. Generate the top 5 teams by attack and defense from the final filtered state
    # 2. Generate the covariance matrix of the final filtered state (Gamma_T)
    # 3. Correlation matrix of the final filtered state
    # 4. filter states over time for top 5 teams by attack and defense
    # print("Final filtered state shape:", filtered_states.particles.x.shape)
    # print("Final log-normalizing constant:", filtered_states.log_normalizing_constant[-1])

    from rbsqmc.src.utils.graphic import plot_all
    plot_all(
        filtered_states=filtered_states,
        augmented_results=model_inputs_rbpf,
        team_id_to_name=team_id_to_name,
        top_n=5,
        save_path="./rbsqmc/outputs/untrained",
        timestamps=data["date"].to_numpy(),
    )

if __name__ == "__main__":
    main()

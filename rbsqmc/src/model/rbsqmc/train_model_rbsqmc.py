"""
Training script for the RB-SQMC model.

This mirrors ``train_model.py`` but replaces the RB-SMC filter with the
RB-SQMC filter from ``model_rbsqmc.py``.  The optimisation loop maximises the
log marginal likelihood estimated by the SQMC filter's log-normalizing
constant, using the same stop-gradient / Fisher-identity gradient approach
as the SMC version.

Usage:
    python -m rbsqmc.src.model.rbsqmc.train_model_rbsqmc

Or directly:
    python rbsqmc/src/model/train_model_rbsqmc.py
"""

import os
import jax
import jax.numpy as jnp
import pandas as pd
import json
import math
import optax
from datetime import datetime
from functools import partial

from rbsqmc.src.data.data import get_results, get_training_data, concat_football_results
from rbsqmc.src.model.rbsqmc.model_rbsqmc import run_filter_sqmc, run_filter_sqmc_logz
from rbsqmc.src.model.rbsmc.model import compute_gamma_trajectory, generate_rbpf_trajectory
from rbsqmc.src.utils.helpers import (
    default_init_params,
    resolve_teams,
    save_params,
    decode_EM_params,
    encode_EM_params,
    log_inverse_wishart_kernel,
)
from rbsqmc.src.utils.type import EMParams, FootballResults, RawEMParams
from rbsqmc.src.utils.graphic import (
    plot_all,
    plot_gradient_norm_curve,
    plot_logmarginal_history_train_test,
)


# ---------------------------------------------------------------------------
# SQMC-based loss and gradient
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("n_particles", "max_goals"))
def loss_fn_sqmc(
    keys: jax.Array,
    model_inputs: FootballResults,
    params: EMParams,
    n_particles: int,
    max_goals: int,
    gamma_0_prior_params: dict | None = None,
):
    """Negative log marginal likelihood, averaged over ``n_reps`` SQMC replicas.

    Same structure as ``optimization.loss_fn`` but uses the SQMC filter
    (``run_filter_sqmc_logz``) instead of the SMC filter.
    """
    logz = jax.vmap(
        lambda k: run_filter_sqmc_logz(
            k, model_inputs, params, n_particles, max_goals
        )
    )(keys)
    loss = -jnp.mean(logz)
    if gamma_0_prior_params is not None:
        loss = loss - gamma_0_prior_params["strength"] * log_inverse_wishart_kernel(
            params.gamma_0, gamma_0_prior_params["scale"], gamma_0_prior_params["dof"]
        )
    return loss


@partial(jax.jit, static_argnames=("n_particles", "max_goals"))
def loss_and_grad_sqmc(
    keys: jax.Array,
    model_inputs: FootballResults,
    raw_params: RawEMParams,
    fixed_mean_0: jax.Array,
    n_particles: int,
    max_goals: int,
    gamma_0_prior_params: dict | None = None,
):
    """Value and gradient of ``loss_fn_sqmc`` w.r.t. unconstrained raw params."""
    def _loss(raw):
        params = decode_EM_params(raw, fixed_mean_0=fixed_mean_0)
        return loss_fn_sqmc(
            keys=keys,
            model_inputs=model_inputs,
            params=params,
            n_particles=n_particles,
            max_goals=max_goals,
            gamma_0_prior_params=gamma_0_prior_params,
        )
    return jax.value_and_grad(_loss)(raw_params)


# ---------------------------------------------------------------------------
# Optimisation loop (SQMC)
# ---------------------------------------------------------------------------

def logmarginal_maximize_sqmc(
    key: jax.Array,
    train_model_inputs: FootballResults,
    test_model_inputs: FootballResults,
    params: EMParams,
    n_particles: int,
    max_goals: int,
    n_epochs: int = 100,
    learning_rate: float = 1e-3,
    n_reps: int = 4,
    gamma_0_prior_params: dict | None = None,
    patience: int | None = None,
):
    """Maximize the log marginal likelihood ``log Z`` with Adam (optax).

    Same structure as ``optimization.logmarginal_maximize`` but uses the
    SQMC filter for both the gradient estimate and the held-out test scoring.
    """
    fixed_mean_0 = params.mean_0
    raw_params = encode_EM_params(params)
    if gamma_0_prior_params is not None and gamma_0_prior_params.get("scale") is None:
        gamma_0_prior_params["scale"] = params.gamma_0

    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=n_epochs,
    )
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(raw_params)

    import time as _time
    logz_history = []
    test_logz_history = []
    grad_norm_history = []
    test_logz = float("nan")
    best_logz = -jnp.inf
    best_test_logz = -jnp.inf
    best_test_epoch = -1
    no_improve_count = 0
    best_params = params
    epoch_times = []
    total_start = _time.perf_counter()

    for epoch in range(n_epochs):
        epoch_start = _time.perf_counter()
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, n_reps)

        loss, grads = loss_and_grad_sqmc(
            keys=keys,
            model_inputs=train_model_inputs,
            raw_params=raw_params,
            fixed_mean_0=fixed_mean_0,
            n_particles=n_particles,
            max_goals=max_goals,
            gamma_0_prior_params=gamma_0_prior_params,
        )
        loss = float(loss)
        logz = -loss
        logz_history.append(logz)
        grad_norm_history.append(float(optax.global_norm(grads)))

        # NaN guard
        grads_finite = jax.tree_util.tree_all(
            jax.tree_util.tree_map(jnp.all, jax.tree_util.tree_map(jnp.isfinite, grads))
        )
        if not (jnp.isfinite(loss) and grads_finite):
            elapsed = _time.perf_counter() - total_start
            print(
                f"[epoch {epoch:4d}] non-finite loss/gradient detected "
                f"(logZ = {logz}); stopping early after {epoch} epochs "
                f"({elapsed:.1f}s elapsed). "
                f"Best finite logZ = {best_logz:.4f}",
                flush=True,
            )
            test_logz_history.append(float("nan"))
            total_sec = _time.perf_counter() - total_start
            print(f"[optimization] stopped early at epoch {epoch} in "
                  f"{total_sec:.1f}s (avg {total_sec / max(epoch + 1, 1):.1f}s/epoch)",
                  flush=True)
            return (
                best_params,
                jnp.asarray(logz_history),
                jnp.asarray(test_logz_history),
                jnp.asarray(grad_norm_history),
            )

        if logz > best_logz:
            best_logz = logz

        # Update parameters with Adam
        updates, opt_state = optimizer.update(grads, opt_state)
        raw_params = optax.apply_updates(raw_params, updates)

        # --- held-out test logZ (forward filter, no gradient) ---
        key, test_key = jax.random.split(key)
        current_params = decode_EM_params(raw_params, fixed_mean_0=fixed_mean_0)
        test_logz = float(
            run_filter_sqmc_logz(
                key=test_key,
                model_inputs=test_model_inputs,
                params=current_params,
                n_particles=n_particles,
                max_goals=max_goals,
            )
        )
        if not math.isfinite(test_logz):
            test_logz = test_logz_history[-1] if test_logz_history else float("nan")
        test_logz_history.append(test_logz)

        # Model-selection checkpoint
        if math.isfinite(test_logz) and test_logz > best_test_logz:
            best_test_logz = test_logz
            best_test_epoch = epoch
            best_params = current_params
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Early stopping
        if patience is not None and no_improve_count >= patience:
            epoch_times.append(_time.perf_counter() - epoch_start)
            total_sec = _time.perf_counter() - total_start
            print(
                f"[epoch {epoch:4d}] early stopping: test logZ not improved for "
                f"{patience} epochs (best test {best_test_logz:.4f} @ epoch "
                f"{best_test_epoch})",
                flush=True,
            )
            print(f"[optimization] early-stopped at epoch {epoch} after "
                  f"{total_sec:.1f}s; best test checkpoint at epoch "
                  f"{best_test_epoch}", flush=True)
            return (
                best_params,
                jnp.asarray(logz_history),
                jnp.asarray(test_logz_history),
                jnp.asarray(grad_norm_history),
            )

        epoch_secs = _time.perf_counter() - epoch_start
        epoch_times.append(epoch_secs)

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            elapsed = _time.perf_counter() - total_start
            avg_sec = sum(epoch_times) / len(epoch_times)
            remaining = avg_sec * (n_epochs - (epoch + 1))
            test_str = (f"  test logZ = {test_logz:.4f}  "
                        f"(best test {best_test_logz:.4f} @ ep {best_test_epoch})"
                        if test_logz_history else "")
            print(
                f"[epoch {epoch:4d}] logZ = {logz:.4f}  (best train {best_logz:.4f})"
                f"{test_str}"
                f"  [{epoch_secs:6.1f}s this epoch, {elapsed:6.1f}s elapsed, "
                f"ETA {remaining:6.1f}s]",
                flush=True,
            )

    total_sec = _time.perf_counter() - total_start
    print(f"[optimization] finished {n_epochs} epochs in {total_sec:.1f}s "
          f"(avg {total_sec / max(n_epochs, 1):.1f}s/epoch); best test logZ "
          f"{best_test_logz:.4f} @ epoch {best_test_epoch}", flush=True)
    return (
        best_params,
        jnp.asarray(logz_history),
        jnp.asarray(test_logz_history),
        jnp.asarray(grad_norm_history),
    )


# ---------------------------------------------------------------------------
# Main training pipeline (mirrors train_model.py)
# ---------------------------------------------------------------------------

def main():
    date_text = datetime.now().strftime("gd_%Y%m%d_%H%M%S")
    output_dir = f"rbsqmc/outputs/train_model_sqmc/{date_text}/"

    cfg = {
        "training_start_date": "1980-01-01",
        "test_start_date": "2024-01-01",
        "prediction_start_date": "2026-06-11",
        "n_particles": 250,          # N
        "max_goals": 8,               # MAX_GOALS
        "seed": 0,                    # PRNG seed
        # optimization
        "n_epochs": 50,
        "learning_rate": 0.05,
        "n_reps": 5,
        "patience": 15,
        # data / output
        "include_friendly": True,
        "teams": "worldcup2026",
        "output_dir": output_dir,
    }
    if not os.path.exists(cfg["output_dir"]):
        os.makedirs(cfg["output_dir"], exist_ok=True)
        print(f"Created output directory: {cfg['output_dir']}")
    else:
        print(f"Output directory already exists: {cfg['output_dir']}")

    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Wrote run config to {os.path.join(output_dir, 'run_config.json')}")

    key = jax.random.PRNGKey(cfg["seed"])
    teams_only = resolve_teams(cfg)
    (train_df, test_df, prediction_df), (train_model_inputs, test_model_inputs, prediction_model_inputs), team_id_to_name = get_training_data(
        train_start_date=cfg["training_start_date"],
        test_start_date=cfg["test_start_date"],
        prediction_start_date=cfg["prediction_start_date"],
        max_goals=cfg["max_goals"],
        include_friendly=cfg["include_friendly"],
        teams_only=teams_only,
    )
    print(f"Extracted training data:")
    print(f"  Training data: {len(train_df)} matches. Training data from {train_df['date'].min().date()} to {train_df['date'].max().date()}")
    print(f"  Test data: {len(test_df)} matches. Test data from {test_df['date'].min().date()} to {test_df['date'].max().date()}")
    print(f"  Prediction data: {len(prediction_df)} matches. Prediction data from {prediction_df['date'].min().date()} to {prediction_df['date'].max().date()}")

    num_teams = len(team_id_to_name)
    params = default_init_params(num_teams=num_teams, team_id_to_name=team_id_to_name)

    ############# LOG MARGINALIZATION OPTIMIZATION (SQMC) ################
    key, opt_key = jax.random.split(key, 2)
    (best_params, train_logz_history, test_logz_history, grad_norm_history) = logmarginal_maximize_sqmc(
        key=opt_key,
        train_model_inputs=train_model_inputs,
        test_model_inputs=test_model_inputs,
        params=params,
        n_particles=cfg["n_particles"],
        max_goals=cfg["max_goals"],
        n_epochs=cfg["n_epochs"],
        learning_rate=cfg["learning_rate"],
        n_reps=cfg["n_reps"],
        gamma_0_prior_params=cfg.get("gamma_0_prior_params"),
        patience=cfg.get("patience"),
    )
    train_logz = [float(v) for v in train_logz_history]
    test_logz = [float(v) for v in test_logz_history]
    grad_norms = [float(v) for v in grad_norm_history]

    # plot the logZ history
    plot_logmarginal_history_train_test(
        train_logz_history=train_logz_history,
        train_match_count=int(train_model_inputs.match_mask.sum()),
        test_logz_history=test_logz_history,
        test_match_count=int(test_model_inputs.match_mask.sum()),
        save_path=os.path.join(cfg["output_dir"], "logmarginal_history_train_test.png"),
    )
    plot_gradient_norm_curve(
        grad_norm_history=grad_norm_history,
        save_path=os.path.join(output_dir, "gradient_norm_curve.png"),
    )
    # save best params to output_dir
    save_params(
        params=best_params,
        path=os.path.join(cfg["output_dir"], "best_params.json")
    )

    ############# FILTERING WITH OPTIMIZED PARAMETERS (SQMC) ################
    # Concatenate train + test so the final filtered state reflects everything
    # through the end of the test split.
    observed_inputs = concat_football_results(train_model_inputs, test_model_inputs)
    key, filter_key = jax.random.split(key, 2)
    final_result, final_model_inputs_rbpf = run_filter_sqmc(
        key=filter_key,
        model_inputs=observed_inputs,
        params=best_params,
        n_particles=cfg["n_particles"],
        max_goals=cfg["max_goals"],
    )

    os.makedirs(os.path.join(cfg["output_dir"], "final_filter"), exist_ok=True)
    # Real match dates for the timeseries x-axis (train then test).
    full_dates = pd.concat([train_df["date"], test_df["date"]]).to_numpy()

    # Build a lightweight state-like object for plot_all compatibility.
    # plot_all expects filtered_states with .particles.x and .log_weights
    # and .log_normalizing_constant.  We wrap the SQMC result dict.
    from rbsqmc.src.utils.type import RBPFState
    from typing import NamedTuple

    class SQMCFilteredStates(NamedTuple):
        particles: RBPFState
        log_weights: jax.Array
        log_normalizing_constant: jax.Array

    filtered_states = SQMCFilteredStates(
        particles=RBPFState(x=final_result['particles_x']),
        log_weights=final_result['log_weights'],
        log_normalizing_constant=final_result['log_normalizing_constant'],
    )

    plot_all(
        filtered_states=filtered_states,
        augmented_results=final_model_inputs_rbpf,
        team_id_to_name=team_id_to_name,
        top_n=10,
        save_path=os.path.join(cfg["output_dir"], "final_filter"),
        timestamps=full_dates,
        params=best_params,
    )

    ############### Run Sequential Prediction for Upcoming Matches ###############
    key, pred_key = jax.random.split(key, 2)
    from rbsqmc.src.model.rbsqmc.predict_rbsqmc import run_sequential_predict_rbsqmc
    from rbsqmc.src.utils.helpers import (
        build_match_predictions,
        save_match_predictions,
    )

    pred_grids, pred_logprobs, daily_logp = run_sequential_predict_rbsqmc(
        key=pred_key,
        observed_inputs=observed_inputs,
        prediction_inputs=prediction_model_inputs,
        params=best_params,
        n_particles=cfg["n_particles"],
        max_goals=cfg["max_goals"],
    )

    # Build per-match prediction dicts and save
    predictions = build_match_predictions(
        all_grids=pred_grids,
        all_logp_actual=pred_logprobs,
        prediction_inputs=prediction_model_inputs,
        team_id_to_name=team_id_to_name,
        max_goals=cfg["max_goals"],
    )
    pred_dir = os.path.join(cfg["output_dir"], "prediction")
    save_match_predictions(
        predictions,
        save_dir=pred_dir,
        max_goals=cfg["max_goals"],
    )

    ############### Observe team states over time (post-prediction) ###############
    from rbsqmc.src.model.rbsmc.train_model_gpu import run_observe

    run_observe(
        cfg=cfg,
        params=best_params,
        output_dir=pred_dir,
    )


if __name__ == "__main__":
    main()

from functools import partial
import math
import os
import json

import jax
import jax.numpy as jnp
import cuthbert
import cuthbertlib
from cuthbertlib.resampling import autodiff
import optax
from datetime import datetime

from rbsqmc.src.utils.type import EMParams, FootballResults, RBPFFootballResults, RawEMParams
from rbsqmc.src.utils.helpers import (
    decode_EM_params,
    encode_EM_params,
    default_init_params,
    save_params,
    resolve_teams,
    log_inverse_wishart_kernel,
)
from rbsqmc.src.data.data import get_results, get_training_data, WORLDCUP_2026_TEAMS
from rbsqmc.src.model.rbsmc.model import init_sample, propagate_sample, _log_potential, compute_gamma_trajectory, generate_rbpf_trajectory

@partial(jax.jit, static_argnames=("n_particles", "max_goals"))
def run_filter_unbiased(
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
        resampling_fn=autodiff.stop_gradient_decorator(
            cuthbertlib.resampling.systematic.resampling
        ),
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

@partial(jax.jit, static_argnames=("n_particles", "max_goals"))
def loss_fn(
    keys : jax.Array,
    model_inputs: FootballResults,
    params : EMParams,
    n_particles : int,
    max_goals: int,
    gamma_0_prior_params: dict | None = None,
):
    """Negative log marginal likelihood, averaged over ``n_reps`` filter replicas.

    Args:
        keys: (n_reps, 2) array of independent PRNG keys.
        model_inputs: FootballResults (raw inputs; RBPF trajectory built inside
            ``run_filter_unbiased``).
        params: EMParams to score.
        gamma_0_prior_params: optional dict with keys ``scale`` (jax.Array),
            ``dof`` (float), and ``strength`` (float). If given, an
            inverse-Wishart prior regularizer ``-strength * log p(gamma_0)``
            is added to the loss, pulling ``gamma_0`` toward ``scale``.
            If None, the original unregularized loss is used.

    Returns:
        ``-mean(log Z)`` plus the (optional) gamma_0 prior regularizer.
    """
    logz = jax.vmap(
        lambda k: run_filter_unbiased(
            k, model_inputs, params, n_particles, max_goals
        )[0].log_normalizing_constant[-1]
    )(keys)
    loss = -jnp.mean(logz)
    if gamma_0_prior_params is not None:
        loss = loss - gamma_0_prior_params["strength"] * log_inverse_wishart_kernel(
            params.gamma_0, gamma_0_prior_params["scale"], gamma_0_prior_params["dof"]
        )
    return loss

@partial(jax.jit, static_argnames=("n_particles", "max_goals"))
def loss_and_grad_raw(
        keys: jax.Array,
        model_inputs: FootballResults,
        raw_params: RawEMParams,
        fixed_mean_0: jax.Array,
        n_particles: int,
        max_goals: int,
        gamma_0_prior_params: dict | None = None,
):
    """Value and gradient of ``loss_fn`` w.r.t. unconstrained raw params.

    ``mean_0`` is fixed (not optimized): the decoded ``EMParams`` always uses
    ``fixed_mean_0``. The gradient flows through ``decode_EM_params`` so it is
    taken with respect to ``raw_params``, which is what the optimizer updates.
    """
    def _loss(raw):
        params = decode_EM_params(raw, fixed_mean_0=fixed_mean_0)
        return loss_fn(
            keys=keys,
            model_inputs=model_inputs,
            params=params,
            n_particles=n_particles,
            max_goals=max_goals,
            gamma_0_prior_params=gamma_0_prior_params,
        )
    return jax.value_and_grad(_loss)(raw_params)

def logmarginal_maximize(
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

    ``mean_0`` is held fixed; the remaining identified parameters
    (``gamma_0``, ``B``, ``kappa``, ``alpha``, ``beta``) are optimized in the
    unconstrained ``RawEMParams`` space via ``encode``/``decode_EM_params``.
    Gradients are the unbiased stop-gradient (Fisher) estimates from
    ``run_filter_unbiased``. The learning rate is cosine-annealed from
    ``learning_rate`` to 0 over ``n_epochs``.

    The **held-out** test split is scored each epoch by running the forward
    filter (no gradient) with the **current** Adam-updated parameters,
    recording ``test_logz_history``. This lets you monitor generalization on
    data that never influences the gradient updates.

    Args:
        test_model_inputs: held-out ``FootballResults`` (test split), used as
            the **validation** signal for model selection / early stopping.
        gamma_0_prior_params: optional dict with keys ``scale`` (jax.Array),
            ``dof`` (float), and ``strength`` (float). If given, an
            inverse-Wishart prior regularizer is added to the loss, pulling
            ``gamma_0`` toward ``scale``. If None, the original unregularized
            loss is used.
        patience: optional int. If given, stop training once the held-out test
            logZ has not improved for ``patience`` consecutive epochs. The
            returned checkpoint is the one with the highest held-out test logZ.

    Returns:
        (best_params, history, test_history, grad_norm_history) where
        ``best_params`` is the decoded ``EMParams`` of the checkpoint with the
        highest held-out **test** logZ (treated as validation; avoids returning
        the overfit final params that maximize training loss), ``history`` is
        the (regularized) negative training loss per epoch, ``test_history`` is
        the forward-filter test logZ per epoch, and ``grad_norm_history`` is the
        global gradient norm per epoch (convergence / instability diagnostic).
    """
    fixed_mean_0 = params.mean_0
    raw_params = encode_EM_params(params)
    if gamma_0_prior_params is not None and gamma_0_prior_params.get("scale") is None:
        gamma_0_prior_params["scale"] = params.gamma_0

    # Cosine-anneal the learning rate from `learning_rate` down to 0 over the
    # training run. This lets Adam take large early steps and shrink to small
    # refinements near the end, improving convergence (matches smoothing_v2).
    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=n_epochs,
    )
    # Use Adam to optimize the unconstrained raw parameters. The gradient is
    optimizer = optax.adam(schedule)
    # Initialize the optimizer state with the raw parameters. The optimizer will
    opt_state = optimizer.init(raw_params)

    import time as _time
    logz_history = []
    test_logz_history = []
    grad_norm_history = []
    test_logz = float("nan")  # last held-out test logZ
    best_logz = -jnp.inf       # best train logZ (display only)
    best_test_logz = -jnp.inf  # best held-out test logZ (model selection)
    best_test_epoch = -1
    no_improve_count = 0       # consecutive epochs without test improvement
    best_params = params
    epoch_times = []  # seconds per epoch, for ETA estimation
    total_start = _time.perf_counter()

    for epoch in range(n_epochs):
        epoch_start = _time.perf_counter()
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, n_reps)

        loss, grads = loss_and_grad_raw(
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

        # NaN guard: stop the run immediately if the loss or gradient norm is
        # not finite (GPU float32 instability), keeping the last valid params.
        # best_params is only updated while logz is finite, so it still holds
        # the best finite parameters seen so far.
        # `grads` is a RawEMParams pytree (NamedTuple), so check finiteness
        # across its leaves rather than passing the whole pytree to jnp.isfinite
        # (which raises TypeError for non-ndarray arguments).
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
            # Record a NaN for the current epoch so the history length still
            # matches the number of epochs actually attempted.
            test_logz_history.append(float("nan"))
            # Keep the best finite params; skip the Adam update.
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
            best_logz = logz  # display-only: best train logZ

        # Update parameters with Adam
        updates, opt_state = optimizer.update(grads, opt_state)
        # Apply updates to raw_params (unconstrained space)
        raw_params = optax.apply_updates(raw_params, updates)

        # --- held-out test logZ (forward filter, no gradient) ---
        # Score the test split with the *current* Adam-updated params, so each
        # entry reflects the exact parameter state at that epoch.
        key, test_key = jax.random.split(key)
        current_params = decode_EM_params(raw_params, fixed_mean_0=fixed_mean_0)
        test_logz = float(
            run_filter_unbiased(
                key=test_key,
                model_inputs=test_model_inputs,
                params=current_params,
                n_particles=n_particles,
                max_goals=max_goals,
            )[0].log_normalizing_constant[-1]
        )
        # The test forward filter can occasionally produce NaN for certain
        # intermediate parameter states (GPU float32 instability in the
        # particle filter).  This does not affect the training loss/grads
        # (which are guarded separately), so carry forward the last finite
        # test logZ instead of recording NaN in the history.
        if not math.isfinite(test_logz):
            test_logz = test_logz_history[-1] if test_logz_history else float("nan")
        test_logz_history.append(test_logz)

        # --- model-selection checkpoint on the held-out test logZ ---
        # best_params is the checkpoint with the highest held-out test logZ,
        # i.e. test is treated as validation for early stopping. This avoids
        # returning the final, overfit parameters that maximize train logZ.
        if math.isfinite(test_logz) and test_logz > best_test_logz:
            best_test_logz = test_logz
            best_test_epoch = epoch
            best_params = current_params
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Early stopping: stop once test logZ has not improved for `patience`
        # consecutive epochs (only when patience is set).
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

def main():
    date_text = datetime.now().strftime("gd_%Y%m%d_%H%M%S")
    output_dir = f"rbsqmc/outputs/optimization/{date_text}/"

    cfg = {
        "start_date": "1900-01-01",
        "split_date": "2025-01-01",  # matches on/before are train, after are test
        "end_date": "2026-01-01",
        "n_particles": 20,          # N (reduced for a quick validation run)
        "max_goals": 8,               # MAX_GOALS
        "seed": 0,                    # PRNG seed
        # optimization
        "n_epochs": 10,               # reduced for a quick validation run
        "learning_rate": 1e-3,
        "n_reps": 5,
        # data / output
        "include_friendly": False,
        "teams": "worldcup2026",
        "output_dir": output_dir,
    }
    key = jax.random.PRNGKey(cfg["seed"])
    teams_only = resolve_teams(cfg)

    # Full dataset (used for baseline/final filtering and plotting).
    # data, model_inputs, team_id_to_name = get_results(
    #     start_date=cfg["start_date"],
    #     end_date=cfg["end_date"],
    #     max_goals=cfg["max_goals"],
    #     include_friendly=cfg["include_friendly"],
    #     teams_only=teams_only,
    # )
    # num_teams = len(team_id_to_name)
    # print(f"Extracted data from {cfg['start_date']} to {cfg['end_date']}, with {num_teams} teams and {len(data)} dates.")
    # print("Number of teams:", num_teams)

    # Train/test split: train = on/before split_date, test = after.
    # Prediction split is pushed beyond end_date so it is empty here.
    (train_df, test_df, _pred_df), (train_inputs, test_inputs, _pred_inputs), team_id_to_name = \
        get_training_data(
            train_start_date=cfg["start_date"],
            test_start_date=cfg["split_date"],
            prediction_start_date=cfg["end_date"],
            prediction_end_date=None,
            max_goals=cfg["max_goals"],
            include_friendly=cfg["include_friendly"],
            teams_only=teams_only,
        )
    print(f"Train: {len(train_df)} dates (<= {cfg['split_date']}); "
          f"Test: {len(test_df)} dates (from {cfg['split_date']}).")

    params = default_init_params(num_teams=num_teams, team_id_to_name=team_id_to_name)

    ############ FILTER ################
    key, filter_key = jax.random.split(key, 2)
    filtered_states, model_inputs_rbpf = run_filter_unbiased(
        key=filter_key,
        model_inputs=model_inputs,
        params=params,
        n_particles=cfg["n_particles"],
        max_goals=cfg["max_goals"],
    )
    baseline_logz = float(filtered_states.log_normalizing_constant[-1])


    ############# LOG MARGINALIZATION OPTIMIZATION ################
    key, opt_key = jax.random.split(key, 2)
    best_params, logz_history, test_logz_history = logmarginal_maximize(
        key=opt_key,
        train_model_inputs=train_inputs,       # train on the train split
        test_model_inputs=test_inputs,   # score the held-out test split each epoch
        params=params,
        n_particles=cfg["n_particles"],
        max_goals=cfg["max_goals"],
        n_epochs=cfg["n_epochs"],
        learning_rate=cfg["learning_rate"],
        n_reps=cfg["n_reps"],
        # gamma_0_prior=gamma_0_prior,
        # gamma_prior_dof=cfg.get("gamma_prior_dof", 5.0),
        # gamma_prior_strength=cfg.get("gamma_prior_strength", 1.0),
    )

    ############## FILTER WITH BEST PARAMETERS ################
    key, final_filter_key = jax.random.split(key, 2)

    final_states, final_model_inputs_rbpf = run_filter_unbiased(
        key=final_filter_key,
        model_inputs=model_inputs,
        params=best_params,
        n_particles=cfg["n_particles"],
        max_goals=cfg["max_goals"],
    )

    ############## SAVE RESULTS ################
    os.makedirs(cfg["output_dir"], exist_ok=True)

    from rbsqmc.src.utils.graphic import (
        # save_filter_states,
        plot_all,
        plot_log_marginal_likelihood_curve,
    )
    plot_all(
        filtered_states=final_states,
        augmented_results=final_model_inputs_rbpf,
        team_id_to_name=team_id_to_name,
        top_n=10,
        save_path=cfg["output_dir"],
        timestamps=data["date"].to_numpy(),
        params=best_params,
    )
    plot_log_marginal_likelihood_curve(
        logz_history,
        save_path=os.path.join(cfg["output_dir"], "optimization_logZ_curve.png"),
    )
    print(f"Smoothing Optimization completed. Saved results to {cfg['output_dir']}")

if __name__ == "__main__":
    main()
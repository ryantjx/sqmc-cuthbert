"""
Run prediction

take in and filter
sequentially generate and store prediction
compare predictions to actual results
update the filter with new match results
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from rbsqmc.src.data.bivariate_poisson import loglik_grid
from rbsqmc.src.data.data import concat_football_results, unpack_football_results
from rbsqmc.src.model.rbsmc.optimization import run_filter_unbiased
from rbsqmc.src.utils.type import EMParams, Matches, RBPFState, FootballResults


def evaluate_match_predictions(predictions: list[dict]) -> dict:
    """Evaluate score and three-outcome forecasts with known results.

    The multiclass Brier score is the unscaled sum over home/draw/away,
    ``sum_c (p_c - o_c)**2``. It ranges from 0 (perfect) to 2 (maximally
    wrong). A uniform three-outcome forecast has score ``2/3`` for every
    match and is included as a fixed reference.

    The input records are enriched in place with per-match evaluation fields.
    Matches whose actual score is negative are retained but not scored.
    """
    brier_scores = []
    log_likelihoods = []
    exact_correct = []
    outcome_correct = []

    for prediction in predictions:
        actual_home = int(prediction["actual_home_score"])
        actual_away = int(prediction["actual_away_score"])
        scored = actual_home >= 0 and actual_away >= 0
        if not scored:
            prediction.update({
                "actual_outcome": None,
                "predicted_outcome": None,
                "brier_score": None,
                "exact_score_correct": None,
                "outcome_correct": None,
            })
            continue

        actual_outcome = (
            "home" if actual_home > actual_away
            else "draw" if actual_home == actual_away
            else "away"
        )
        probabilities = {
            "home": float(prediction["prob_home_win"]),
            "draw": float(prediction["prob_draw"]),
            "away": float(prediction["prob_away_win"]),
        }
        predicted_outcome = max(probabilities, key=probabilities.get)
        brier = sum(
            (probability - float(outcome == actual_outcome)) ** 2
            for outcome, probability in probabilities.items()
        )
        exact = (
            int(prediction["predicted_home_score"]) == actual_home
            and int(prediction["predicted_away_score"]) == actual_away
        )
        outcome_hit = predicted_outcome == actual_outcome

        prediction.update({
            "actual_outcome": actual_outcome,
            "predicted_outcome": predicted_outcome,
            "brier_score": float(brier),
            "exact_score_correct": bool(exact),
            "outcome_correct": bool(outcome_hit),
        })
        brier_scores.append(float(brier))
        log_likelihoods.append(float(prediction["log_likelihood"]))
        exact_correct.append(float(exact))
        outcome_correct.append(float(outcome_hit))

    n_scored = len(brier_scores)
    uniform_brier = 2.0 / 3.0
    mean_brier = float(sum(brier_scores) / n_scored) if n_scored else None
    return {
        "n_predictions": len(predictions),
        "n_scored": n_scored,
        "brier_score_definition": "sum_over_home_draw_away",
        "mean_brier_score": mean_brier,
        "uniform_reference_brier_score": uniform_brier,
        "brier_skill_score_vs_uniform": (
            float(1.0 - mean_brier / uniform_brier)
            if mean_brier is not None
            else None
        ),
        "total_log_likelihood": (
            float(sum(log_likelihoods)) if n_scored else None
        ),
        "mean_log_likelihood": (
            float(sum(log_likelihoods) / n_scored) if n_scored else None
        ),
        "exact_score_accuracy": (
            float(sum(exact_correct) / n_scored) if n_scored else None
        ),
        "outcome_accuracy": (
            float(sum(outcome_correct) / n_scored) if n_scored else None
        ),
    }

@jax.jit(static_argnames=("max_goals",))
def predict_match_score(
    particles_x: jax.Array,     # (N, num_teams, 2)
    log_weights: jax.Array,     # (N,)
    home_id: int,
    away_id: int,
    alpha: float,
    beta: float,
    max_goals: int,
    scale: float = 1.0,
):
    """Weighted posterior predictive score grid (G x G), normalized to sum to 1."""
    grid, _ = predict_match_score_with_mass(
        particles_x, log_weights, home_id, away_id,
        alpha, beta, max_goals, scale,
    )
    return grid


@jax.jit(static_argnames=("max_goals",))
def predict_match_score_with_mass(
    particles_x: jax.Array,     # (N, num_teams, 2)
    log_weights: jax.Array,     # (N,)
    home_id: int,
    away_id: int,
    alpha: float,
    beta: float,
    max_goals: int,
    scale: float = 1.0,
):
    """Weighted posterior predictive score grid plus its raw (unnormalized) mass.

    Returns ``(grid, raw_mass)`` where ``grid`` is the ``(G, G)`` score grid
    normalized to sum to one and ``raw_mass`` is the unnormalized total
    likelihood mass ``exp(logsumexp(logp))``. The raw mass is recorded *before*
    normalization so it can diagnose score-grid truncation; the sum of an
    already-normalized grid cannot.
    """
    # (N, G, G) log-likelihood grids, one per particle
    log_grid = jax.vmap(
        lambda p: loglik_grid(
            p[home_id], p[away_id], alpha=alpha, beta=beta,
            max_goals=max_goals, scale=scale)
    )(particles_x)

    # Weighted average of likelihoods: logsumexp over particles with the
    # unnormalized log-weights, normalized in log space. Normalizing directly
    # in log space makes the result invariant to adding any constant to all
    # log weights (a shift that must not change the normalized mixture).
    log_w = log_weights - logsumexp(log_weights)  # (N,) normalized log-weights
    logp = logsumexp(log_grid + log_w[:, None, None], axis=0)  # (G, G)

    log_mass = logsumexp(logp)
    grid = jnp.exp(logp - log_mass)
    raw_mass = jnp.exp(log_mass)
    return grid, raw_mass

# ---------------------------------------------------------------------------
# Sequential predict -> update -> compare driver
# ---------------------------------------------------------------------------
def run_sequential_predict(
    key: jax.Array,
    observed_inputs: FootballResults,
    prediction_inputs: FootballResults,
    params: EMParams,
    n_particles: int,
    max_goals: int,
):
    """Unpack to match-level steps, then run compiled SMC prediction."""
    return _run_sequential_predict_jitted(
        key=key,
        observed_inputs=unpack_football_results(observed_inputs),
        prediction_inputs=unpack_football_results(prediction_inputs),
        params=params,
        n_particles=n_particles,
        max_goals=max_goals,
    )


@partial(jax.jit, static_argnames=("n_particles", "max_goals"))
def _run_sequential_predict_jitted(
    key: jax.Array,
    observed_inputs: FootballResults,
    prediction_inputs: FootballResults,
    params: EMParams,
    n_particles: int,
    max_goals: int,
):
    """One-step-ahead prediction over the prediction window.

    Returns per-match predicted grids and predictive log-probabilities of the
    actual scores.
    """
    # ---- 1. Filter over the FULL concatenated sequence --------------------
    # concat_football_results fixes the timestamp_prev boundary and pads M.
    full_inputs = concat_football_results(observed_inputs, prediction_inputs)
    filtered_states, _ = run_filter_unbiased(
        key=key,
        model_inputs=full_inputs,
        params=params,
        n_particles=n_particles,
        max_goals=max_goals,
    )
    # filtered_states has leading dimension T_full + 1 (index 0 is initial).
    # At index t+1, particle positions have already been resampled and
    # propagated for match t, but only their weights use match t's score.
    # Therefore those positions with uniform pre-likelihood weights form the
    # one-step predictive particle approximation for match t.

    G = max_goals + 1
    # In full-sequence coords, prediction day p (0-based) is full day
    # t = n_obs_days + p. Its predictive state is at filtered index t.
    n_obs_days = observed_inputs.timestamp.shape[0]  # days in the observed sequence
    pred_start = n_obs_days  # full-sequence day index of first prediction day

    def scan_body(_, pred_day):
        t = pred_start + pred_day            # full-sequence day index
        x_predictive = filtered_states.particles.x[t + 1]  # (N, teams, 2)
        log_weights_predictive = jnp.zeros_like(
            filtered_states.log_weights[t + 1]
        )

        home_id = prediction_inputs.matches.home_id[pred_day]   # (M,)
        away_id = prediction_inputs.matches.away_id[pred_day]
        valid = prediction_inputs.match_mask[pred_day]
        yh = prediction_inputs.matches.home_score[pred_day]
        ya = prediction_inputs.matches.away_score[pred_day]

        def one_match(h, a, v, sh, sa):
            grid = predict_match_score(x_predictive, log_weights_predictive, h, a,
                                       params.alpha, params.beta, max_goals)
            grid = jnp.where(v, grid, jnp.zeros((G, G)))
            # predictive log-prob of the ACTUAL score (compare pred vs actual)
            score_is_known = (
                v
                & (sh >= 0)
                & (sa >= 0)
                & (sh <= max_goals)
                & (sa <= max_goals)
            )
            safe_home_score = jnp.clip(sh, 0, max_goals)
            safe_away_score = jnp.clip(sa, 0, max_goals)
            logp_actual = jnp.where(
                score_is_known,
                jnp.log(grid[safe_home_score, safe_away_score] + 1e-12),
                0.0,
            )
            return grid, logp_actual

        grids, logp_actual = jax.vmap(one_match)(home_id, away_id, valid, yh, ya)
        # grids: (M, G, G); logp_actual: (M,)
        day_logp = jnp.sum(logp_actual)            # predictive log-score for the day
        return None, (grids, logp_actual, day_logp)

    n_pred_days = prediction_inputs.timestamp.shape[0]
    _, (all_grids, all_logp_actual, day_logp) = jax.lax.scan(
        scan_body, None, jnp.arange(n_pred_days)
    )
    # all_grids:      (P, M, G, G)  predicted grid per match per day
    # all_logp_actual:(P, M)        predictive log-prob of actual score per match
    # day_logp:       (P,)          total predictive log-prob per day
    return all_grids, all_logp_actual, day_logp


def main():
    pass

if __name__ == "__main__":
    main()

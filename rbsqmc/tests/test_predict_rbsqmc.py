"""Tests for SQMC sequential prediction and forecast evaluation."""

import jax
import jax.numpy as jnp
import numpy as np

from rbsqmc.src.model.rbsmc.predict import (
    evaluate_match_predictions,
    predict_match_score,
    predict_match_score_with_mass,
    run_sequential_predict,
)
from rbsqmc.src.model.rbsqmc.predict_rbsqmc import (
    predict_from_sqmc_history,
    run_sequential_predict_rbsqmc,
)
from rbsqmc.src.model.rbsqmc.model_rbsqmc import run_filter_sqmc
from rbsqmc.src.utils.type import EMParams, FootballResults, Matches


def _inputs(timestamp, timestamp_prev, home, away, home_score, away_score):
    return FootballResults(
        date=jnp.asarray(timestamp) + 10,
        timestamp=jnp.asarray(timestamp),
        timestamp_prev=jnp.asarray(timestamp_prev),
        matches=Matches(
            home_id=jnp.asarray(home)[:, None],
            away_id=jnp.asarray(away)[:, None],
            home_score=jnp.asarray(home_score)[:, None],
            away_score=jnp.asarray(away_score)[:, None],
        ),
        match_mask=jnp.ones((len(timestamp), 1), dtype=bool),
    )


def _params():
    return EMParams(
        mean_0=jnp.zeros((4, 2)),
        gamma_0=jnp.eye(4),
        B=jnp.array([[1.0, 0.2], [0.2, 1.0]]),
        kappa=jnp.array(0.001),
        alpha=jnp.array(0.2),
        beta=jnp.array(-4.0),
    )


def test_sqmc_prediction_matches_smc_output_contract():
    observed = _inputs([2], [0], [0], [1], [1], [0])
    prediction = _inputs([3, 3], [2, 3], [2, 0], [3, 2], [2, -1], [1, -1])
    arguments = {
        "key": jax.random.PRNGKey(0),
        "observed_inputs": observed,
        "prediction_inputs": prediction,
        "params": _params(),
        "n_particles": 8,
        "max_goals": 3,
    }

    smc = run_sequential_predict(**arguments)
    sqmc = run_sequential_predict_rbsqmc(**arguments)
    jax.block_until_ready(sqmc[0])

    assert smc[0].shape == sqmc[0].shape == (2, 1, 4, 4)
    assert smc[1].shape == sqmc[1].shape == (2, 1)
    assert smc[2].shape == sqmc[2].shape == (2,)
    np.testing.assert_allclose(np.asarray(smc[0]).sum(axis=(-2, -1)), 1.0)
    np.testing.assert_allclose(np.asarray(sqmc[0]).sum(axis=(-2, -1)), 1.0)
    assert np.isfinite(np.asarray(smc[0])).all()
    assert np.isfinite(np.asarray(sqmc[0])).all()
    # The second fixture has no known score and must not receive a bogus log score.
    assert float(smc[1][1, 0]) == 0.0
    assert float(sqmc[1][1, 0]) == 0.0


def test_prediction_evaluation_includes_multiclass_brier_reference():
    predictions = [
        {
            "actual_home_score": 1,
            "actual_away_score": 0,
            "predicted_home_score": 1,
            "predicted_away_score": 0,
            "prob_home_win": 1.0,
            "prob_draw": 0.0,
            "prob_away_win": 0.0,
            "log_likelihood": 0.0,
        },
        {
            "actual_home_score": 0,
            "actual_away_score": 1,
            "predicted_home_score": 1,
            "predicted_away_score": 0,
            "prob_home_win": 1.0,
            "prob_draw": 0.0,
            "prob_away_win": 0.0,
            "log_likelihood": -2.0,
        },
        {
            "actual_home_score": -1,
            "actual_away_score": -1,
            "predicted_home_score": 0,
            "predicted_away_score": 0,
            "prob_home_win": 0.3,
            "prob_draw": 0.4,
            "prob_away_win": 0.3,
            "log_likelihood": 0.0,
        },
    ]

    evaluation = evaluate_match_predictions(predictions)

    assert evaluation["n_predictions"] == 3
    assert evaluation["n_scored"] == 2
    assert evaluation["mean_brier_score"] == 1.0
    assert evaluation["uniform_reference_brier_score"] == 2.0 / 3.0
    assert evaluation["brier_skill_score_vs_uniform"] == -0.5
    assert evaluation["outcome_accuracy"] == 0.5
    assert predictions[0]["brier_score"] == 0.0
    assert predictions[1]["brier_score"] == 2.0
    assert predictions[2]["brier_score"] is None


# ---------------------------------------------------------------------------
# Repaired-pipeline regression tests (scale consistency, shared history,
# current-fixture scale, no score leakage, axes/fixed-state ordering).
# ---------------------------------------------------------------------------


def _run_filter(observed, prediction, params, n_particles, max_goals, scales):
    """Run the filter once over observed+prediction and return the history."""
    from rbsqmc.src.data.data import concat_football_results
    full = concat_football_results(observed, prediction)
    result, _ = run_filter_sqmc(
        key=jax.random.PRNGKey(0),
        model_inputs=full,
        params=params,
        n_particles=n_particles,
        max_goals=max_goals,
        match_scales=scales,
    )
    return result


def test_current_fixture_scale_matches_independent_loglik_grid():
    """A one-particle forecast equals the independently normalized loglik_grid
    at a non-unit scale, including the raw mass."""
    from rbsqmc.src.data.bivariate_poisson import loglik_grid
    from jax.scipy.special import logsumexp

    params = _params()
    max_goals = 3
    scale = 2.0
    # One particle with distinct attack/defence for teams 0 and 1.
    particles_x = jnp.array([[[0.5, 0.2], [0.1, 0.4], [0.0, 0.0], [0.0, 0.0]]])
    log_weights = jnp.zeros(1)

    grid, raw_mass = predict_match_score_with_mass(
        particles_x, log_weights, 0, 1, params.alpha, params.beta, max_goals, scale,
    )
    # Independent reference: single-particle loglik_grid at the same scale.
    log_grid = loglik_grid(
        particles_x[0, 0], particles_x[0, 1],
        alpha=params.alpha, beta=params.beta, max_goals=max_goals, scale=scale,
    )
    log_mass = logsumexp(log_grid)
    ref_grid = jnp.exp(log_grid - log_mass)
    ref_mass = jnp.exp(log_mass)

    np.testing.assert_allclose(np.asarray(grid), np.asarray(ref_grid), atol=1e-6)
    np.testing.assert_allclose(float(raw_mass), float(ref_mass), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(grid).sum(), 1.0, atol=1e-6)


def test_log_weight_shift_invariance():
    """Adding any finite constant to all log weights must leave both the grid
    and the raw mass unchanged (normalized weights are shift-invariant)."""
    params = _params()
    max_goals = 3
    # Two particles with distinct strengths and non-uniform weights 0.99/0.01.
    particles_x = jnp.array([
        [[0.5, 0.2], [0.1, 0.4], [0.0, 0.0], [0.0, 0.0]],
        [[0.1, 0.1], [0.6, 0.3], [0.0, 0.0], [0.0, 0.0]],
    ])
    log_weights = jnp.log(jnp.array([0.99, 0.01]))

    grid_a, mass_a = predict_match_score_with_mass(
        particles_x, log_weights, 0, 1, params.alpha, params.beta, max_goals, 1.0,
    )
    grid_b, mass_b = predict_match_score_with_mass(
        particles_x, log_weights - 100.0, 0, 1, params.alpha, params.beta, max_goals, 1.0,
    )
    np.testing.assert_allclose(np.asarray(grid_a), np.asarray(grid_b), atol=1e-6)
    np.testing.assert_allclose(float(mass_a), float(mass_b), rtol=1e-6)


def test_predict_match_score_scale_changes_grid():
    """The trailing scale argument changes the score grid (scale is not ignored)."""
    params = _params()
    max_goals = 3
    particles_x = jnp.array([[[0.5, 0.2], [0.1, 0.4], [0.0, 0.0], [0.0, 0.0]]])
    log_weights = jnp.zeros(1)
    g1 = predict_match_score(particles_x, log_weights, 0, 1, params.alpha, params.beta, max_goals, 1.0)
    g2 = predict_match_score(particles_x, log_weights, 0, 1, params.alpha, params.beta, max_goals, 2.0)
    assert not np.allclose(np.asarray(g1), np.asarray(g2))


def test_swap_teams_transposes_grid():
    """Swapping home/away transposes the score grid (axis convention)."""
    params = _params()
    max_goals = 3
    particles_x = jnp.array([[[0.5, 0.2], [0.1, 0.4], [0.0, 0.0], [0.0, 0.0]]])
    log_weights = jnp.zeros(1)
    g_ab = predict_match_score(particles_x, log_weights, 0, 1, params.alpha, params.beta, max_goals, 1.0)
    g_ba = predict_match_score(particles_x, log_weights, 1, 0, params.alpha, params.beta, max_goals, 1.0)
    np.testing.assert_allclose(np.asarray(g_ab), np.asarray(g_ba).T, atol=1e-6)


def test_fixed_state_higher_total_team_higher_win_prob():
    """For one fixed particle, the team with greater total strength has the
    greater win probability. This is a fixed-state assertion, not one imposed
    on posterior mean totals."""
    params = _params()
    max_goals = 3
    # Team 0 total = 0.7, team 1 total = 0.5 -> team 0 stronger.
    particles_x = jnp.array([[[0.5, 0.2], [0.1, 0.4], [0.0, 0.0], [0.0, 0.0]]])
    log_weights = jnp.zeros(1)
    grid = predict_match_score(particles_x, log_weights, 0, 1, params.alpha, params.beta, max_goals, 1.0)
    G = max_goals + 1
    ii, jj = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    prob_home = float(np.asarray(grid)[ii > jj].sum())
    prob_away = float(np.asarray(grid)[ii < jj].sum())
    assert prob_home > prob_away


def test_no_score_leakage_into_own_forecast():
    """Changing the current score leaves its own forecast unchanged but changes
    the posterior; a later forecast can then change. Includes sequential
    matches on the same date."""
    params = _params()
    max_goals = 3
    n_particles = 8
    observed = _inputs([2], [0], [0], [1], [1], [0])
    # Two prediction matches on the same date (t=3), then one on the next date
    # (t=4). The score of the first same-date match must not leak into its own
    # forecast, but must change the next-date forecast.
    pred_a = _inputs([3, 3, 4], [2, 3, 3], [2, 0, 1], [3, 2, 0], [2, 1, 0], [1, 0, 1])
    pred_b = _inputs([3, 3, 4], [2, 3, 3], [2, 0, 1], [3, 2, 0], [2, 0, 0], [1, 1, 1])

    scales = jnp.ones((observed.timestamp.shape[0] + 3, 1))
    result_a = _run_filter(observed, pred_a, params, n_particles, max_goals, scales)
    result_b = _run_filter(observed, pred_b, params, n_particles, max_goals, scales)

    out_a = predict_from_sqmc_history(
        result_a, pred_a, observed.timestamp.shape[0], params, max_goals,
        jnp.ones((3, 1)),
    )
    out_b = predict_from_sqmc_history(
        result_b, pred_b, observed.timestamp.shape[0], params, max_goals,
        jnp.ones((3, 1)),
    )
    # The first same-date forecast must be identical (its own score is not
    # assimilated into its own forecast).
    np.testing.assert_allclose(np.asarray(out_a[0][0]), np.asarray(out_b[0][0]), atol=1e-6)
    # The second same-date forecast is also unaffected (same pre-likelihood
    # particles), but the next-date forecast propagates from the changed
    # posterior and can differ.
    np.testing.assert_allclose(np.asarray(out_a[0][1]), np.asarray(out_b[0][1]), atol=1e-6)
    assert not np.allclose(np.asarray(out_a[0][2]), np.asarray(out_b[0][2]))


def test_predict_from_sqmc_history_equals_sequential_driver():
    """The extracted history helper reproduces the sequential driver's grids
    when both use the same filter and scales."""
    params = _params()
    max_goals = 3
    n_particles = 8
    observed = _inputs([2], [0], [0], [1], [1], [0])
    prediction = _inputs([3, 3], [2, 3], [2, 0], [3, 2], [2, -1], [1, -1])
    observed_scales = jnp.ones((1, 1))
    prediction_scales = jnp.ones((2, 1))

    seq = run_sequential_predict_rbsqmc(
        key=jax.random.PRNGKey(0),
        observed_inputs=observed,
        prediction_inputs=prediction,
        params=params,
        n_particles=n_particles,
        max_goals=max_goals,
        observed_scales=observed_scales,
        prediction_scales=prediction_scales,
    )
    result = _run_filter(
        observed, prediction, params, n_particles, max_goals,
        jnp.concatenate([observed_scales, prediction_scales]),
    )
    hist = predict_from_sqmc_history(
        result, prediction, observed.timestamp.shape[0], params, max_goals,
        prediction_scales,
    )
    np.testing.assert_allclose(np.asarray(seq[0]), np.asarray(hist[0]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(seq[1]), np.asarray(hist[1]), atol=1e-6)


"""Focused regression tests for the RB-SQMC forward filter."""

import jax
import jax.numpy as jnp
import numpy as np

import rbsqmc.src.model.rbsqmc.model_rbsqmc as model_rbsqmc
from rbsqmc.src.model.rbsqmc.model_rbsqmc import (
    _differentiable_resampled_log_weights,
    _log_match_potential_batched,
    hilbert_resample,
    propagate_match_transform,
    run_filter_sqmc,
)
from rbsqmc.src.utils.type import (
    EMParams,
    FootballResults,
    Matches,
)


def _params(num_teams: int) -> EMParams:
    return EMParams(
        mean_0=jnp.zeros((num_teams, 2)),
        gamma_0=jnp.eye(num_teams),
        B=jnp.eye(2),
        kappa=jnp.array(0.01),
        alpha=jnp.array(0.2),
        beta=jnp.array(-4.0),
    )


def _results(
    home_id,
    away_id,
    home_score,
    away_score,
    match_mask,
) -> FootballResults:
    return FootballResults(
        date=jnp.array([0]),
        timestamp=jnp.array([0.0]),
        timestamp_prev=jnp.array([0.0]),
        matches=Matches(
            home_id=jnp.asarray([home_id]),
            away_id=jnp.asarray([away_id]),
            home_score=jnp.asarray([home_score]),
            away_score=jnp.asarray([away_score]),
        ),
        match_mask=jnp.asarray([match_mask]),
    )


def test_hilbert_resample_preserves_u_v_point_pairing():
    particles = jnp.zeros((3, 2, 2))
    log_weights = jnp.full((3,), -jnp.log(3.0))
    points = jnp.array(
        [
            [0.8, 8.0, 80.0, 800.0, 8000.0],
            [0.2, 2.0, 20.0, 200.0, 2000.0],
            [0.5, 5.0, 50.0, 500.0, 5000.0],
        ]
    )

    _, v_sorted = hilbert_resample(
        particles_x=particles,
        log_weights=log_weights,
        rqmc_points=points,
        obs_indices=jnp.array([0, 1]),
        n_particles=3,
    )

    np.testing.assert_array_equal(
        np.asarray(v_sorted),
        np.asarray(points[jnp.array([1, 2, 0]), 1:]),
    )


def test_log_normalizer_has_exactly_one_particle_average():
    n_particles = 8
    inputs = _results([0], [1], [1], [0], [True])
    params = _params(2)

    result, _ = run_filter_sqmc(
        jax.random.PRNGKey(0),
        inputs,
        params,
        n_particles=n_particles,
        max_goals=8,
    )
    log_likelihood = _log_match_potential_batched(
        particles_x=result["particles_x"][-1],
        home_id=jnp.array(0),
        away_id=jnp.array(1),
        home_score=jnp.array(1),
        away_score=jnp.array(0),
        alpha=params.alpha,
        beta=params.beta,
        max_goals=8,
    )
    expected = jax.scipy.special.logsumexp(log_likelihood) - jnp.log(
        n_particles
    )

    np.testing.assert_allclose(
        np.asarray(result["log_normalizing_constant"][-1]),
        np.asarray(expected),
        rtol=0.0,
        atol=1e-12,
    )


def test_disjoint_matches_receive_independent_rqmc_coordinates():
    inputs = _results(
        home_id=[0, 2],
        away_id=[1, 3],
        home_score=[1, 0],
        away_score=[0, 1],
        match_mask=[True, True],
    )

    result, _ = run_filter_sqmc(
        jax.random.PRNGKey(0),
        inputs,
        _params(4),
        n_particles=8,
        max_goals=8,
    )
    particles = np.asarray(result["particles_x"][-1])

    # With identity team covariance, reusing one v point for both matches made
    # these arrays exactly equal. Independent per-match points must break that.
    assert not np.array_equal(particles[:, 0], particles[:, 2])
    assert not np.array_equal(particles[:, 1], particles[:, 3])
    assert np.isfinite(particles).all()


def test_gaussian_quantile_clips_rqmc_endpoints():
    v_t = jnp.array(
        [
            [0.0, 0.25, 0.75, 1.0],
            [1.0, 0.75, 0.25, 0.0],
        ],
        dtype=jnp.float64,
    )
    propagated = propagate_match_transform(
        particles_x=jnp.zeros((2, 2, 2)),
        B=jnp.eye(2),
        kalman_gain=jnp.eye(2),
        home_id=jnp.array(0),
        away_id=jnp.array(1),
        gamma_observed=jnp.eye(2),
        v_t=v_t,
        n_particles=2,
    )

    assert bool(jnp.all(jnp.isfinite(propagated)))


def test_differentiable_resampling_is_uniform_but_retains_score_gradient():
    log_weights = jnp.array([-2.0, -1.0, -0.5])
    ancestors = jnp.array([2, 2, 0])

    resampled = _differentiable_resampled_log_weights(
        log_weights,
        ancestors,
        n_particles=3,
    )
    gradient = jax.grad(
        lambda weights: jnp.sum(
            _differentiable_resampled_log_weights(
                weights,
                ancestors,
                n_particles=3,
            )
        )
    )(log_weights)

    np.testing.assert_allclose(
        np.asarray(resampled),
        np.full(3, -np.log(3.0)),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(np.asarray(gradient), np.array([1.0, 0.0, 2.0]))


def test_main_runs_one_pass_with_standard_model_parameters(monkeypatch, capsys):
    inputs = _results([0], [1], [1], [0], [True])
    calls = {}
    params = object()
    augmented_inputs = object()
    expected_result = {
        "particles_x": jnp.zeros((2, 10, 2, 2)),
        "log_weights": jnp.full((2, 10), -jnp.log(10.0)),
        "log_normalizing_constant": jnp.array([0.0, -2.5]),
    }

    def fake_get_results(**kwargs):
        calls["data"] = kwargs
        return [object()], inputs, {0: "A", 1: "B"}

    def fake_default_init_params(**kwargs):
        calls["params"] = kwargs
        return params

    def fake_run_filter_sqmc(**kwargs):
        calls["filter"] = kwargs
        return expected_result, augmented_inputs

    monkeypatch.setattr(model_rbsqmc, "get_results", fake_get_results)
    monkeypatch.setattr(
        model_rbsqmc,
        "default_init_params",
        fake_default_init_params,
    )
    monkeypatch.setattr(
        model_rbsqmc,
        "run_filter_sqmc",
        fake_run_filter_sqmc,
    )

    result, augmented = model_rbsqmc.main()

    assert result is expected_result
    assert augmented is augmented_inputs
    assert calls["data"] == {
        "start_date": "1980-01-01",
        "end_date": "2026-01-01",
        "max_goals": 8,
        "include_friendly": False,
        "teams_only": model_rbsqmc.WORLDCUP_2026_TEAMS,
        "download": False,
    }
    assert calls["params"] == {
        "num_teams": 2,
        "team_id_to_name": {0: "A", 1: "B"},
    }
    assert calls["filter"]["params"] is params
    assert calls["filter"]["n_particles"] == 10
    assert calls["filter"]["max_goals"] == 8
    np.testing.assert_array_equal(
        np.asarray(calls["filter"]["key"]),
        np.asarray(jax.random.PRNGKey(42)),
    )
    assert "final log Z=-2.500000" in capsys.readouterr().out

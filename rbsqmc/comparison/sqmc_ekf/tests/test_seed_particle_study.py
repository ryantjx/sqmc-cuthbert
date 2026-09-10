"""Regression checks for the F8 study's mass and posterior summaries."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from rbsqmc.comparison.sqmc_ekf.scripts import seed_particle_study as study


def test_fixture_exports_raw_mass_and_weighted_posterior(monkeypatch):
    # The score grid contains only 0-0: its normalized mass is always one,
    # while its actual probability can be checked analytically.
    x = jnp.array([[[-2., 0.], [0., 0.]],
                   [[1., 0.], [0., 0.]],
                   [[4., 0.], [0., 0.]]])
    lw = jnp.log(jnp.array([.1, .7, .2]))
    result = dict(particles_x=jnp.stack([x, x]),
                  log_weights=jnp.stack([jnp.zeros(3), lw]))
    monkeypatch.setattr(study, "run_filter_sqmc", lambda *a, **kw: (result, None))
    dataset = SimpleNamespace(inputs=SimpleNamespace(friendly=jnp.array([False])),
                              sqmc=None, frame=pd.DataFrame([dict(home_id=0, away_id=1)]))
    params = dict(model=SimpleNamespace(alpha=0., beta=-4.), friendly_scale=2.)
    entry = study.evaluate_fixture(dataset, params, dict(max_goals=0),
                                   jax.random.PRNGKey(0), 3, 0, 0, 1.)
    expected_mass = np.exp(-np.exp(np.array([-2., 1., 4.])) - 1 - np.exp(-4.)).mean()
    assert entry["raw_grid_mass"] == pytest.approx(expected_mass, rel=1e-6)
    assert entry["raw_grid_mass"] < .5
    assert entry["outcome_probabilities"]["draw"] == pytest.approx(1.)
    pred = entry["total_strength_difference"]
    post = entry["post_total_strength_difference"]
    assert pred["mean"] == pytest.approx(1.)
    assert post["mean"] == pytest.approx(1.3)
    assert [post[q] for q in ("q05", "q50", "q95")] == [-2., 1., 4.]
    assert post["fraction_positive"] == pytest.approx(.9)
    means = entry["posterior_means"]
    assert post["mean"] == pytest.approx(
        means["home_attack"] + means["home_defence"]
        - means["away_attack"] - means["away_defence"])
    summary = study.summarize([entry])
    assert summary["raw_grid_mass"]["mean"] == entry["raw_grid_mass"]
    assert summary["posterior_total_difference"]["mean"] == post["mean"]


def test_weighted_distribution_zero_weights_and_shift_invariance():
    x = np.zeros((3, 2, 2))
    x[:, 0, 0] = [-100., 2., 9.]
    lw = np.array([-np.inf, np.log(.99), np.log(.01)])
    actual = study._total_diff_distribution(x, 0, 1, lw)
    shifted = study._total_diff_distribution(x, 0, 1, lw - 1000.)
    assert actual == pytest.approx(shifted)
    assert actual["mean"] == pytest.approx(2.07)
    assert [actual[q] for q in ("q05", "q50", "q95")] == [2., 2., 2.]
    assert actual["fraction_positive"] == pytest.approx(1.)


@pytest.mark.parametrize("weights", [[-np.inf] * 3, [0., np.nan, 0.], [0., np.inf, 0.], [0.]])
def test_invalid_posterior_weights_rejected(weights):
    with pytest.raises(ValueError, match="Posterior log weights"):
        study._total_diff_distribution(np.zeros((3, 2, 2)), 0, 1, weights)

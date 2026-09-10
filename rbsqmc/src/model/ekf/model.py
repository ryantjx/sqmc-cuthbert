"""Factorial EKF, adapted from cuthberto-carlos (Apache-2.0).

Reference revision: f79147e82f8cfa37c1df9a35d2508954bf3f5d97.
Uses Cuthbert 0.0.14's explicit initialization and scalar OU mean reversion.
"""

from functools import partial
from typing import NamedTuple

import ghq
import jax
import jax.numpy as jnp
from cuthbert.factorial import filter as factorial_filter
from cuthbert.factorial.gaussian import build_factorializer
from cuthbert.gaussian import moments

from rbsqmc.src.data.bivariate_poisson import _loglik_grid_loglambdas


class MatchInputs(NamedTuple):
    home: jax.Array
    away: jax.Array
    score: jax.Array
    friendly: jax.Array
    timestamp: jax.Array
    previous: jax.Array  # Last appearance of each playing team, shape (T, 2).


def positive(raw):
    return jax.nn.softplus(raw) + 1e-6


def inverse_positive(value):
    return jnp.log(jnp.expm1(jnp.asarray(value) - 1e-6))


def constrain(raw):
    """Reference parameterization: shared variance and attack–defence correlation."""
    sd = positive(raw["init_sd"])
    corr = 0.99 * jnp.tanh(raw["init_corr"])
    cov = sd**2 * jnp.array([[1.0, corr], [corr, 1.0]])
    return dict(init_mean=jnp.zeros(2), init_cov=cov,
                init_chol_cov=jnp.linalg.cholesky(cov),
                kappa=positive(raw["kappa"]), alpha=raw["alpha"],
                beta=raw["beta"], friendly_scale=positive(raw["friendly_scale"]))


def initial_raw(cov, alpha=0.2, beta=-4.0, kappa=0.001, friendly_scale=2.0):
    return dict(init_sd=inverse_positive(jnp.sqrt(cov[0, 0])),
                init_corr=jnp.arctanh(cov[0, 1] / cov[0, 0] / 0.99),
                kappa=inverse_positive(kappa), alpha=jnp.asarray(alpha),
                beta=jnp.asarray(beta), friendly_scale=inverse_positive(friendly_scale))


def build(params, num_teams, match_scale=1.0):
    def init(_):
        return (jnp.broadcast_to(params["init_mean"], (num_teams, 2)),
                jnp.broadcast_to(params["init_chol_cov"], (num_teams, 2, 2)))

    def dynamics(state, match):
        elapsed = match.timestamp - match.previous
        phi = jnp.exp(-params["kappa"] * elapsed)
        # sqrt(0) has an undefined derivative: branch before taking the root.
        # A same-time match has exactly zero process noise, not added jitter.
        variance = -jnp.expm1(-2 * params["kappa"] * elapsed)
        roots = jnp.where(elapsed > 0, jnp.sqrt(jnp.where(elapsed > 0, variance, 1)), 0)
        chol = jax.scipy.linalg.block_diag(
            roots[0] * params["init_chol_cov"], roots[1] * params["init_chol_cov"])
        mean = jnp.tile(params["init_mean"], 2)
        def conditional(x):
            return mean + jnp.repeat(phi, 2) * (x - mean), chol
        return conditional, state.mean

    def observation(state, match):
        # Friendlies use the learned friendly_scale; all other matches use the
        # configured match_scale baseline, matching the SQMC observation model.
        scale = jnp.where(match.friendly, params["friendly_scale"], match_scale)
        def conditional(x):
            l1 = jnp.exp(params["alpha"] + (x[0] - x[3]) / scale)
            l2 = jnp.exp(params["alpha"] + (x[2] - x[1]) / scale)
            l3 = jnp.exp(params["beta"])
            return (jnp.array([l1 + l3, l2 + l3]),
                    jnp.linalg.cholesky(jnp.array([[l1 + l3, l3], [l3, l2 + l3]])))
        # Cuthbert's documented missing-observation convention skips assimilation.
        score = jnp.where(jnp.all(match.score >= 0), match.score, jnp.nan)
        return conditional, state.mean, score

    return (moments.build_filter(init, dynamics, observation),
            build_factorializer(lambda match: jnp.array([match.home, match.away])))


def run_filter(inputs, params, num_teams, match_scale=1.0):
    """Return team marginals and cumulative Gaussian approximate logZ, prior first."""
    filter_obj, factorializer = build(params, num_teams, match_scale)
    first = jax.tree.map(lambda x: x[0], inputs)
    initial = filter_obj.init_prepare(first)
    # Cuthbert joins the two teams, updates jointly, then retains only their
    # separate marginals. No fictitious match is inserted at initialization.
    history = factorial_filter(filter_obj, factorializer, inputs, initial,
                               output_factorial=True)
    return dict(mean=history.mean,
                cov=history.chol_cov @ jnp.swapaxes(history.chol_cov, -1, -2),
                logz=history.log_normalizing_constant)


def propagate(mean, cov, elapsed, params):
    phi = jnp.exp(-params["kappa"] * elapsed)
    return (params["init_mean"] + phi[..., None] * (mean - params["init_mean"]),
            phi[..., None, None]**2 * cov
            + (-jnp.expm1(-2 * params["kappa"] * elapsed))[..., None, None] * params["init_cov"])


@partial(jax.jit, static_argnames=("max_goals", "degree"))
def predict_match(mean, cov, alpha, beta, scale, max_goals=8, degree=32):
    """Integrate in the two-dimensional Gaussian log-rate space; return raw mass."""
    transform = jnp.array([[1., 0., 0., -1.], [0., -1., 1., 0.]]) / scale
    joint_cov = jax.scipy.linalg.block_diag(cov[0], cov[1])
    rate_mean = alpha + transform @ mean.reshape(4)
    rate_cov = transform @ joint_cov @ transform.T
    return ghq.multivariate(
        lambda rates: jnp.exp(_loglik_grid_loglambdas(rates[0], rates[1], beta, max_goals)),
        rate_mean, rate_cov, degree=degree)


def sequential_predict(inputs, history, params, start, max_goals=8, degree=32, match_scale=1.0):
    """Forecast from the previous state; the current score cannot enter its grid."""
    def one(_, index):
        pair = jnp.array([inputs.home[index], inputs.away[index]])
        mean, cov = propagate(history["mean"][index, pair], history["cov"][index, pair],
                              inputs.timestamp[index] - inputs.previous[index], params)
        scale = jnp.where(inputs.friendly[index], params["friendly_scale"], match_scale)
        grid = predict_match(mean, cov, params["alpha"], params["beta"], scale, max_goals, degree)
        return None, grid
    return jax.lax.scan(one, None, jnp.arange(start, len(inputs.timestamp)))[1]


def synchronized_moments(inputs, history, params, num_teams, match_scale=1.0):
    """Propagate reporting copies to a common date without changing filter logZ."""
    def step(last, index):
        last = last.at[inputs.home[index]].set(inputs.timestamp[index])
        last = last.at[inputs.away[index]].set(inputs.timestamp[index])
        mean, cov = propagate(history["mean"][index + 1], history["cov"][index + 1],
                              inputs.timestamp[index] - last, params)
        return last, (mean, cov)
    return jax.lax.scan(step, jnp.zeros(num_teams), jnp.arange(len(inputs.timestamp)))[1]

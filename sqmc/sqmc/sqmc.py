"""
Sequential Quasi-Monte Carlo (SQMC) implementation in JAX compatible with `cuthbert` and `cuthbertlib`.

generate N RQMC points of 1+d dimension
Hilbert sort previous N particles to get Hilbert-ordered weights
Sort RQMC by the first coordinate
select ancestors by inverse CDF using first coordinates and Hilbert-ordered weights
Propagate ancestors through a deterministic function propagate_transform (instead of propagate_sample, which is stochastic) using d coordinates
Reweight observation
update log-normalization constant

https://github.com/nchopin/particles/blob/master/particles/core.py#L317
https://github.com/state-space-models/cuthbert/blob/main/cuthbert/smc/particle_filter.py
https://github.com/state-space-models/cuthbert/blob/main/cuthbert/smc/types.py


[1] Gerber, M., & Chopin, N. (2015). Sequential quasi-Monte Carlo. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 77(3), 509-579.
[2] Gerber, M., & Chopin, N. (2019). Sequential quasi-Monte Carlo smoothing. Journal of the American Statistical Association, 114(525), 1550-1567.
[3] Chopin, N., & Papaspiliopoulos, O. (2020). An Introduction to Sequential Monte Carlo. Springer, Springer Series in Statistics.
[4]
"""

from sqmc.hilbert_sort.hilbert_sort import hilbert_sort
from sqmc.qmc.qmc import QMC, Sobol, Halton

from functools import partial
from jax import tree

import cuthbert
import cuthbertlib
from cuthbert.smc.particle_filter import ParticleFilterState
from cuthbert.inference import Filter
from cuthbertlib.types import Array, ArrayTree, ArrayTreeLike, KeyArray, ScalarArray
from cuthbert.smc.types import InitSample, LogPotential, PropagateSample

import jax
import jax.numpy as jnp
from jax import random

from typing import Protocol, cast

def resample_from_uniform(sorted_uniforms, logits):
    """Inverse-CDF ancestor selection from sorted RQMC uniforms and Hilbert-ordered logits.

    Uses a pure-JAX ``searchsorted`` implementation instead of
    ``cuthbertlib.resampling.utils.inverse_cdf``, whose numba-compiled CPU
    callback fails under numba 0.66.0 (a typing regression in
    ``numba/core/typing/npydecl.py``). The pure-JAX path is what the GPU
    branch of ``inverse_cdf`` uses and is correct for sorted uniforms.

    Adopts half-open cumulative intervals (task 7): for ``u`` in ``[0, 1)``,
    selects the first cumulative probability strictly greater than ``u`` via
    a right-sided ``searchsorted``. The cumulative weights are normalised by
    their final value so the upper endpoint is exactly one, which keeps a
    trailing zero-weight plateau at one. A rounded input of exactly one is
    mapped to ``nextafter(1, 0)``.

    Returns ``(idx, logits_out)``. ``idx`` indexes into the *Hilbert-ordered*
    particle ordering, so the caller must map back with ``h_order[idx]``.
    """
    # Accept a trailing singleton dimension (the filter carries (N, 1) log
    # weights); squeeze it so the cumulative search is one-dimensional.
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits[:, 0]
    if logits.ndim != 1 or sorted_uniforms.ndim != 1:
        raise ValueError("logits and sorted_uniforms must be one-dimensional.")
    if logits.shape[0] == 0:
        raise ValueError("logits must be nonempty.")

    weights = jax.nn.softmax(logits)
    cdf = jnp.cumsum(weights)
    cdf = cdf / cdf[-1]
    one = jnp.asarray(1, dtype=sorted_uniforms.dtype)
    zero = jnp.asarray(0, dtype=sorted_uniforms.dtype)
    # Explicit policy: tolerate a rounded input of exactly one, but reject
    # values outside [0, 1] (the caller is responsible for finite uniforms).
    u = jnp.minimum(sorted_uniforms, jnp.nextafter(one, zero))
    idx = jnp.searchsorted(cdf, u, side="right", method="sort")
    logits_out = jnp.zeros_like(sorted_uniforms)
    return idx, logits_out


def _sample_points(qmc: QMC, key: KeyArray, n: int) -> jnp.ndarray:
    """Sample ``n`` QMC points, scrambling with ``key`` when enabled.

    When ``qmc.scramble`` is True, a fresh scrambled engine is constructed
    from ``key`` so each call produces an independent randomization starting at
    index zero (task 6). When False, the captured engine's deterministic
    sequence is used and ``key`` is ignored.
    """
    if qmc.scramble:
        engine = Sobol(
            d=qmc.d,
            scramble=True,
            key=key,
            dtype=qmc.dtype,
            start_index=0,
        )
        return cast(jnp.ndarray, engine.sample(n))
    return cast(jnp.ndarray, qmc.sample(n))


class InitTransform(Protocol):
    def __call__(self, u: Array, model_inputs: ArrayTreeLike) -> ArrayTree:
        ...

class PropagateTransform(Protocol):
    def __call__(self, u: Array, state: ArrayTreeLike, model_inputs: ArrayTreeLike) -> ArrayTree:
        ...

def build_filter(
    init_transform: InitTransform,
    propagate_transform: PropagateTransform,
    log_potential: LogPotential,
    n_filter_particles: int,
    qmc: QMC
) -> Filter:
    # init_sample uses the key to scramble the RQMC points when enabled.
    # The QMC point set has dimension du + 1 (resampling + state); init only
    # needs the first du coordinates.
    def init_sample(key, model_inputs):
        u = _sample_points(qmc, key, n_filter_particles)  # (N, du + 1)
        u = u[:, : qmc.d - 1]                        # slice (N, du) for the initial state
        return jax.vmap(init_transform, (0, None))(u, model_inputs)

    return Filter(
        init_prepare=partial(
            init_prepare, 
            init_sample=init_sample,
            n_filter_particles=n_filter_particles
        ),
        filter_prepare=partial(
            filter_prepare, 
            init_transform=init_transform,
            qmc=qmc,
            n_filter_particles=n_filter_particles
        ),
        filter_combine=partial(
            filter_combine,
            propagate_transform=propagate_transform,
            log_potential=log_potential,
            qmc=qmc
        ),
        associative=False,
    )
# def filter_prepare() -> ParticleFilterState:
#     pass

def _hilbert_sort_particles(particles: ArrayTree) -> Array:
    """Hilbert-sort a particle tree, using the first leaf as the sort key.

    ``hilbert_sort`` requires a single ``(n,)`` or ``(n, d)`` array, but
    ``ParticleFilterState.particles`` is an ``ArrayTree``. For state-space
    models the state is usually a single array; when the tree has multiple
    leaves we sort on the first leaf and apply the resulting permutation to
    every leaf.
    """
    leaves = tree.leaves(particles)
    if not leaves:
        raise ValueError("particles tree is empty.")
    return hilbert_sort(leaves[0])


def init_prepare(
    model_inputs: ArrayTreeLike,
    init_sample : InitSample,
    n_filter_particles: int,
    key: KeyArray | None = None,
) -> ParticleFilterState:
    """Prepare the initial state for the SQMC filter.

    Unlike the stochastic particle filter, ``init_sample`` here is the
    deterministic ``init_transform`` mapped over the full RQMC point set, so
    it already returns the complete ``(N, du)`` batch. We call it directly
    rather than ``vmap``-ing a single-particle sampler over N keys.
    """
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)
    if key is None:
        raise ValueError("A JAX PRNG key must be provided.")

    # Sample the full batch of N particles from the fixed RQMC points.
    particles = init_sample(key, model_inputs)

    # Weight
    log_weights = jnp.zeros(n_filter_particles)

    # Compute the log normalizing constant
    log_normalizing_constant = jax.nn.logsumexp(log_weights) - jnp.log(
        n_filter_particles
    )

    return ParticleFilterState(
        key=key,
        particles=particles,
        log_weights=log_weights,
        ancestor_indices=jnp.arange(n_filter_particles),
        model_inputs=model_inputs,
        log_normalizing_constant=log_normalizing_constant,
    )


def filter_prepare(
    model_inputs: ArrayTreeLike,
    init_transform : InitTransform,
    qmc: QMC,
    n_filter_particles: int,
    key: KeyArray | None = None,
) -> ParticleFilterState:
    """Prepare an empty state for the current SQMC step.

    Only the particle shapes are needed here; the actual particles are
    produced by ``filter_combine``. The per-particle shape is derived by
    applying ``jax.eval_shape`` to the deterministic ``init_transform`` with an
    abstract uniform-coordinate input, so no QMC points are consumed (task 5).
    """
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)
    if key is None:
        raise ValueError("A JAX PRNG key must be provided.")

    # Derive the per-particle shape without sampling or scrambling.
    uniform_spec = jax.ShapeDtypeStruct((qmc.d - 1,), qmc.dtype)
    particle_spec = jax.eval_shape(
        lambda u: init_transform(u, model_inputs), uniform_spec
    )
    particles = tree.map(
        lambda spec: jnp.empty((n_filter_particles,) + spec.shape, spec.dtype),
        particle_spec,
    )
    log_weights = jnp.zeros(n_filter_particles)
    
    # log normalizing constant starts from the weights 
    # identical to jnp.array = 0 because log_weights are all zeros, so logsumexp(log_weights) = log(N)
    log_normalizing_constant = jax.nn.logsumexp(log_weights) - jnp.log(
        n_filter_particles
    )
    return ParticleFilterState(
        key=key,
        particles=particles,
        log_weights=log_weights,
        ancestor_indices=jnp.arange(n_filter_particles),
        model_inputs=model_inputs,
        log_normalizing_constant=log_normalizing_constant,
    )

def filter_combine(
    state_1: ParticleFilterState,
    state_2: ParticleFilterState,
    propagate_transform: PropagateTransform,
    log_potential: LogPotential,
    qmc: QMC
) -> ParticleFilterState:
    N = state_1.log_weights.shape[0]

    # 1. RQMC points of dimension 1 + d, scrambled with the carried key (task 6)
    u = _sample_points(qmc, state_1.key, N)      # (N, 1 + d)

    # 2. Sort RQMC by first coordinate
    tau = jnp.argsort(u[:, 0])

    # 3. Hilbert-sort previous particles -> Hilbert-ordered weights
    h_order = _hilbert_sort_particles(state_1.particles)
    hilbert_log_weights = state_1.log_weights[h_order]

    # 4. Select ancestors via inverse CDF (RQMC first coords + Hilbert-ordered weights)
    idx, _ = resample_from_uniform(u[tau, 0], hilbert_log_weights)
    ancestor_indices = h_order[idx]        # map back to original particle indices
    ancestors = tree.map(lambda x: x[ancestor_indices], state_1.particles)

    # 5. Deterministic propagation using the remaining d coordinates
    v = u[tau, 1:]                         # (N, d)
    next_particles = jax.vmap(propagate_transform, (0, 0, None))(
        v, ancestors, state_2.model_inputs
    )

    # 6. Reweight (weights reset to 0 after resampling)
    log_potentials = jax.vmap(log_potential, (0, 0, None))(
        ancestors, next_particles, state_2.model_inputs
    )
    next_log_weights = log_potentials

    # 7. Update log-normalising constant
    logsum_weights = jax.nn.logsumexp(next_log_weights)
    log_normalizing_constant_incr = logsum_weights - jnp.log(N)
    log_normalizing_constant = (
        log_normalizing_constant_incr + state_1.log_normalizing_constant
    )

    # Carry the prepared key forward so the next observation uses a distinct
    # randomization (task 6 key schedule). ``state_1.key`` was consumed to
    # scramble this step's points; ``state_2.key`` is the fresh key supplied by
    # ``filter_prepare`` for the current observation.
    return ParticleFilterState(
        state_2.key,
        next_particles,
        next_log_weights,
        ancestor_indices,
        state_2.model_inputs,
        log_normalizing_constant,
    )

def main():
    """Build an SQMC filter and run it on a simple 1D random-walk model."""
    def _configure_platform() -> str:
        """Select the JAX backend: GPU when available, otherwise CPU.

        Returns the name of the selected platform. This must be called before any
        JAX computation so the backend is chosen up front.
        """
        try:
            if jax.devices(backend="gpu"):
                jax.config.update("jax_platform_name", "gpu")
                return "gpu"
        except Exception:
            pass
        jax.config.update("jax_platform_name", "cpu")
        return "cpu"

    jax.config.update("jax_enable_x64", True)
    platform = _configure_platform()
    print(f"Running SQMC on {platform}.")

    # --- Model: 1D random walk with Gaussian observation noise ---
    #   x_t = x_{t-1} + sigma_x * Z_t,   Z_t ~ N(0, 1)
    #   y_t = x_t + sigma_y * E_t,       E_t ~ N(0, 1)
    sigma_x = 0.5
    sigma_y = 1.0
    n_particles = 1024
    n_steps = 20

    # Deterministic transforms (inverse-CDF / quantile function).
    def init_transform(u, model_inputs):
        # x_0 ~ N(0, 1)
        return jax.scipy.stats.norm.ppf(u)

    def propagate_transform(u, state, model_inputs):
        # x_t = x_{t-1} + sigma_x * Phi^{-1}(u)
        return state + sigma_x * jax.scipy.stats.norm.ppf(u)

    def log_potential(state_prev, state, model_inputs):
        # log N(y_t | x_t, sigma_y^2)
        y = model_inputs["y"]
        return -0.5 * ((y - state) / sigma_y) ** 2 - jnp.log(sigma_y) - 0.5 * jnp.log(
            2 * jnp.pi
        )

    # Build the SQMC filter. State dimension du = 1, so the QMC point set
    # needs du + 1 = 2 coordinates (1 for resampling, 1 for the state).
    qmc = Sobol(d=2)
    filter_ = build_filter(
        init_transform=init_transform,
        propagate_transform=propagate_transform,
        log_potential=log_potential,
        n_filter_particles=n_particles,
        qmc=qmc,
    )

    # Generate observations from the model, sampling the initial state from
    # the prior X_0 ~ N(0, 1) (task 9).
    key = random.PRNGKey(0)
    x_true = jax.random.normal(key, ())
    observations = []
    for t in range(n_steps):
        key = random.fold_in(key, t)
        x_true = x_true + sigma_x * random.normal(key, ())
        key = random.fold_in(key, t + 1000)
        observations.append(x_true + sigma_y * random.normal(key, ()))
    observations = jnp.array(observations)

    # Run the filter, updating on every observation including the first
    # (task 9). Split the trajectory key into separate initialization and
    # observation streams so no key is reused (task 6).
    trajectory_key = random.PRNGKey(0)
    init_key, steps_key = random.split(trajectory_key)
    step_keys = random.split(steps_key, n_steps + 1)

    state = filter_.init_prepare({"y": observations[0]}, key=init_key)
    state = state._replace(key=step_keys[0])
    for observation, next_key in zip(observations, step_keys[1:]):
        prepared = filter_.filter_prepare({"y": observation}, key=next_key)
        state = filter_.filter_combine(state, prepared)

    weights = jax.nn.softmax(state.log_weights)
    final_mean = jnp.tensordot(weights, state.particles, axes=(0, 0))

    print(f"Estimated log-likelihood: {state.log_normalizing_constant:.4f}")
    print(f"Final mean state estimate: {final_mean:.4f}")


if __name__ == "__main__":
    main()